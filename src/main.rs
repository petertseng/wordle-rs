use std::collections::HashMap;

// Since full analysis fixes the first guess,
// the remaining number of guesses is only 5.
const MAX_TURNS: u8 = 5;

fn score(guess: &str, answer: &str) -> u8 {
    let mut sum = 0_u8;
    let mut freq = [0_u8; 26];
    let mut n = 2_u16;
    for (g, a) in guess.chars().zip(answer.chars()) {
        if g == a {
            // max is 162.
            // 162 * 3 overflows u8,
            // but 162 * 3 will never be added.
            sum += u8::try_from(n).unwrap();
        } else {
            freq[a as usize - 'a' as usize] += 1;
        }
        n *= 3;
    }
    let mut n = 1_u8;
    for (g, a) in guess.chars().zip(answer.chars()) {
        let gord = g as usize - 'a' as usize;
        if g != a && freq[gord] > 0 {
            sum += n;
            freq[gord] -= 1;
        }
        n *= 3;
    }
    sum
}

fn group_by_score<'a>(
    guess: &str,
    candidates: &[&'a str],
    limit: usize,
) -> Option<HashMap<u8, Vec<&'a str>>> {
    let mut by_score = HashMap::new();
    for &cand in candidates {
        let s = score(guess, cand);
        let v = by_score.entry(s).or_insert_with(Vec::new);
        if v.len() == limit {
            return None;
        }
        v.push(cand);
    }
    Some(by_score)
}

fn group_by_score_and_sort<'a, 'b>(
    guesses: &[&'a str],
    answers: &[&'b str],
    limit: usize,
) -> Vec<(&'a str, HashMap<u8, Vec<&'b str>>)> {
    let answer_set: std::collections::HashSet<_> = answers.iter().collect();
    let mut words_and_group: Vec<_> = guesses
        .iter()
        .filter_map(|&guess| group_by_score(guess, answers, limit).map(|g| (guess, g)))
        .collect();

    words_and_group.sort_by_cached_key(|(w, g)| {
        (
            g.iter().map(|(_, v)| v.len() * v.len()).sum::<usize>(),
            if answer_set.contains(w) { 0 } else { 1 },
        )
    });

    words_and_group
}

// for hard mode
//
// bitfield indicating which positions are green
// (relying on being paired with answer list to indicate exactly which letter goes in each)
// vector of (letter, required number of that letter)
// combine_restriction will sort the vector by letter.
type GuessRestriction = (u8, Vec<(char, u8)>);

// confirmed by modifying localStorage (where answer is stored):
// if you get two yellows on a single letter,
// hard mode guesses must have two of that letter.
//
// It's okay to re-guess yellow letters in same position, however!
// And it's okay to re-guess grey letters.
fn filter_guesses<'a>(
    guess: &str,
    status: u8,
    guesses: &[&'a str],
) -> (Vec<&'a str>, GuessRestriction) {
    let mut require_count = HashMap::new();
    let mut require_pos = HashMap::new();
    let mut require_pos_bits = 0;
    let mut curstat = status;
    for (i, g) in guess.chars().enumerate() {
        let trit = curstat % 3;
        if trit > 0 {
            *require_count.entry(g).or_insert(0) += 1;
            if trit == 2 {
                require_pos.insert(i, g);
                require_pos_bits |= 1 << i;
            }
        }

        curstat /= 3;
        if curstat == 0 {
            break;
        }
    }
    let gs = guesses.iter().filter(|gs| {
        let mut have_count = HashMap::new();
        for (i, g) in gs.chars().enumerate() {
            *have_count.entry(g).or_insert(0) += 1;
            if let Some(&c) = require_pos.get(&i) {
                if c != g {
                    return false;
                }
            }
        }
        for (c, count) in require_count.iter() {
            if have_count.get(c).unwrap_or(&0) < count {
                return false;
            }
        }
        true
    });
    (
        gs.cloned().collect(),
        (require_pos_bits, require_count.into_iter().collect()),
    )
}

fn combine_restriction(r1: &GuessRestriction, r2: &GuessRestriction) -> GuessRestriction {
    use std::collections::BTreeMap;
    let (poses1, letters1) = r1;
    let (poses2, letters2) = r2;
    let mut letters: BTreeMap<_, _> = letters1.iter().cloned().collect();
    for (letter, freq2) in letters2 {
        let freq = letters.entry(*letter).or_insert(0);
        *freq = std::cmp::max(*freq, *freq2);
    }

    (poses1 | poses2, letters.into_iter().collect())
}

// DecisionTree used only by rate_starter
#[derive(Debug)]
struct DecisionTree<'a> {
    guess: &'a str,
    branch: HashMap<u8, DecisionTree<'a>>,
}

impl<'a> DecisionTree<'a> {
    fn from_seed(
        starter: &'a str,
        answers: &[&'a str],
        guessables: &[&'a str],
        hard: bool,
    ) -> Self {
        let groups = group_by_score(starter, answers, answers.len()).unwrap();
        Self {
            guess: starter,
            branch: Self::branches(&groups, starter, guessables, hard),
        }
    }

    fn pick_min_mean(answers: &[&'a str], guessables: &[&'a str], hard: bool) -> Self {
        if answers.len() == 1 {
            Self {
                guess: answers[0],
                branch: HashMap::new(),
            }
        } else {
            let sorted = group_by_score_and_sort(guessables, answers, answers.len());
            let (word, ref groups) = sorted[0];

            Self {
                guess: word,
                branch: Self::branches(groups, word, guessables, hard),
            }
        }
    }

    fn branches(
        groups: &HashMap<u8, Vec<&'a str>>,
        guess: &str,
        guessables: &[&'a str],
        hard: bool,
    ) -> HashMap<u8, DecisionTree<'a>> {
        groups
            .iter()
            .map(|(&stat, words)| {
                if hard {
                    let (new_guessables, _) = filter_guesses(guess, stat, guessables);
                    (stat, Self::pick_min_mean(words, &new_guessables, hard))
                } else {
                    (stat, Self::pick_min_mean(words, guessables, hard))
                }
            })
            .collect()
    }
}

fn rate_starter(
    starter: &str,
    answers: &[&str],
    guessables: &[&str],
    hard: bool,
) -> (u32, f64, std::collections::BTreeMap<u8, u32>) {
    let mut count = HashMap::new();
    let tree = DecisionTree::from_seed(starter, answers, guessables, hard);
    for ans in answers {
        let mut guesses = 1;
        let mut current = &tree;
        while current.guess != *ans {
            guesses += 1;
            current = &current.branch[&score(current.guess, ans)];
        }
        *count.entry(guesses).or_insert(0) += 1;
    }
    let total = count.iter().map(|(&n, freq)| u32::from(n) * freq).sum();
    (
        total,
        f64::from(total) / answers.len() as f64,
        count.into_iter().collect(),
    )
}

// number of turns, depth, word
type Best<'a> = Option<(u32, u8, &'a str)>;

// The innermost cache is keyed by answers,
// and just tells the best answer for those answers.
//
// However, if I cached a None response at 2 guesses left,
// I'd still want to check if it's possible at 3 guesses left.
// Therefore, caches also need to be keyed by guesses_left.
//
// And finally, caches need to be keyed by guess restriction.
// Otherwise, you may get a cache hit and use a word you're not allowed to.
//
// TODO: There are some improvements to be explored here.
// * The inner cache key is the full list of answers,
//   but might it instead be possible to store some sort of summary like GuessRestriction?
//   For example, five characters (green letter at that position, if any),
//   plus an array of 26, where each element says the range of possible count for that letter.
//   It is possible that doing this is faster than copying the whole list.
// * Instead of just the current GuessRestriction, maybe store a list of GuessRestriction,
//   those being the ones successively imposed by guesses.
//   Could walk backwards from most restrictive to least, checking for cache entries.
//   If an entry is found and its word is legal, can just use it.
fn best_time<'a>(
    guesses_left: u8,
    answers: &[&'a str],
    guessables: &[&'a str],
    hard: Option<GuessRestriction>,
    gr_caches: &mut HashMap<Option<GuessRestriction>, HashMap<u8, HashMap<Vec<&'a str>, Best<'a>>>>,
) -> Best<'a> {
    if guesses_left == 0 {
        return None;
    }
    assert!(!answers.is_empty());
    if answers.len() == 1 {
        return Some((1, 1, answers[0]));
    }
    if answers.len() == 2 {
        return if guesses_left >= 2 {
            Some((3, 2, answers[0]))
        } else {
            None
        };
    }

    let mut best = u32::MAX;
    let mut best_word = "";
    let mut best_depth = 1;

    if let Some(caches) = gr_caches.get(&hard) {
        if let Some(cache) = caches.get(&guesses_left) {
            if cache.contains_key(answers) {
                return cache[answers];
            }
        }
        // It's acceptable to look in caches of more guesses_left.
        // We do get cache hits this way,
        // though haven't confirmed whether time savings outweigh lookup time.
        for n in (guesses_left + 1)..=MAX_TURNS {
            if let Some(cache) = caches.get(&n) {
                if let Some(prev_entry) = cache.get(answers) {
                    if let Some((_, prev_depth, _)) = prev_entry {
                        // Previously solved.
                        // Need to make sure there are enough guesses to use.
                        if *prev_depth <= guesses_left {
                            return *prev_entry;
                        } else {
                            // If N guesses wasn't usable,
                            // no way N+1 guesses will be.
                            break;
                        }
                    } else {
                        // Previously found unsolvable.
                        return None;
                    }
                }
            }
        }
        // What about caches of fewer guesses_left?
        // They may have needed to compromise on average time to fit in guesses_left,
        // but their result can still be used as a baseline.
        for delta in 1..guesses_left {
            let n = guesses_left - delta;
            if let Some(cache) = caches.get(&n) {
                if let Some(prev_entry) = cache.get(answers) {
                    if let Some((prev_best, prev_depth, prev_word)) = prev_entry {
                        // Previously solved. Set as baseline.
                        best = *prev_best;
                        best_depth = *prev_depth;
                        best_word = prev_word;
                        break;
                    } else {
                        // Previously found unsolvable.
                        // Do nothing as it might be solvable with more.
                    }
                }
            }
        }
    }

    let limit = if guesses_left <= 2 {
        1
    } else if guesses_left == 3 {
        // Which word splits the guesses into the most buckets?
        // If it can split the guesses into N buckets,
        // but there are more than N words in a bucket,
        // there will be > 1 word in a bucket next turn.
        guessables
            .iter()
            .filter_map(|g| group_by_score(g, answers, answers.len() - 1).map(|g| g.len()))
            .max()
            .unwrap()
    } else {
        answers.len() - 1
    };

    let sorted_words = if limit > 1 {
        // perfects take priority; only look for others if there are none.
        let perfects = group_by_score_and_sort(guessables, answers, 1);
        if perfects.is_empty() {
            group_by_score_and_sort(guessables, answers, limit)
        } else {
            perfects
        }
    } else {
        assert_ne!(limit, 0);
        group_by_score_and_sort(guessables, answers, 1)
    };

    for (guess, groups) in sorted_words {
        // for any one group, the best case is:
        // one guess splits the space exactly into buckets of one each,
        // and is an answer itself.
        // That guess takes one guess, the rest take two.
        // Thus the best case is that a group of size n takes takes 2n - 1 guesses.
        let potentials = groups.iter().map(|(&stat, ans)| {
            if stat == 242 {
                0
            } else {
                2 * u32::try_from(ans.len()).unwrap() - 1
            }
        });
        let mut potential = potentials.sum::<u32>();
        if potential >= best {
            continue;
        }

        let mut sum = 0;
        let mut depth = 1;
        let mut group_ok = true;

        let mut sorted_groups: Vec<_> = groups.into_iter().collect();
        sorted_groups.sort_unstable_by_key(|(_, g)| g.len());
        // Interesting question: whether smallest first or largest first is better.
        //
        // My current experimentation seems to indicate:
        // Smallest first makes groups that take not much time take a little more time,
        // and groups that take a really long time take much less time.
        // Since I want groups that take a really long time to not take so long,
        // let's do smallest first.
        //sorted_groups.reverse();
        let sorted_groups = sorted_groups;

        for (status, new_answers) in sorted_groups {
            if status == 242 {
                continue;
            }
            let new_best = if let Some(ref r1) = hard {
                let (new_guessables, r2) = filter_guesses(guess, status, guessables);
                best_time(
                    guesses_left - 1,
                    &new_answers,
                    &new_guessables,
                    Some(combine_restriction(r1, &r2)),
                    gr_caches,
                )
            } else {
                best_time(guesses_left - 1, &new_answers, guessables, None, gr_caches)
            };
            if let Some((new_total, new_depth, _)) = new_best {
                sum += new_total;
                potential -= 2 * u32::try_from(new_answers.len()).unwrap() - 1;
                potential += new_total;
                if potential >= best {
                    group_ok = false;
                    break;
                }
                depth = std::cmp::max(depth, new_depth + 1);
            } else {
                // If any group is unsolvable, the whole thing is unsolvable.
                group_ok = false;
                break;
            }
        }
        if group_ok && sum < best {
            assert_eq!(sum, potential);
            best = sum;
            best_word = guess;
            best_depth = depth;
        }
    }

    let ans = if !best_word.is_empty() {
        Some((
            best + u32::try_from(answers.len()).unwrap(),
            best_depth,
            best_word,
        ))
    } else {
        None
    };
    gr_caches
        .entry(hard)
        .or_insert_with(HashMap::new)
        .entry(guesses_left)
        .or_insert_with(HashMap::new)
        .insert(answers.to_owned(), ans);
    ans
}

fn stat_colour(guess: &str, stat: u8) -> String {
    let mut s = String::new();
    let mut stat = stat;
    for c in guess.chars() {
        match stat % 3 {
            0 => s += "\x1b[0m",
            1 => s += "\x1b[1;33m",
            2 => s += "\x1b[1;32m",
            _ => unreachable!(),
        }
        s.push(c);
        stat /= 3;
    }
    s += "\x1b[0m";
    s
}

fn fully_explore_starter(starter: &str, answers: &[&str], words: &[&str], hard: bool) {
    use std::time::Instant;

    let mut movessum = 0;
    let mut lensum = 0;

    let groups = group_by_score(starter, answers, answers.len()).unwrap();
    let mut groups: Vec<_> = groups.iter().collect();
    groups.sort_unstable_by_key(|(s, g)| (g.len(), *s));
    let groups = groups;

    for (i, &(&stat, group_answers)) in groups.iter().enumerate() {
        // There is no point keeping cache between groups,
        // because groups are all disjoint.
        //
        // There could be a point in keeping cache between starters,
        // but in typical usage one starter is run for each program execution.
        let mut cache = HashMap::new();

        let (filtered, _) = filter_guesses(starter, stat, words);
        let guessables = if hard { &filtered } else { words };
        let tstart = Instant::now();
        lensum += u32::try_from(group_answers.len()).unwrap();
        eprintln!(
            "start  {col} {ii:3}/{glen:3} {stat:3} {anslen:3} {guesslen:5}",
            col = stat_colour(starter, stat),
            ii = i + 1,
            glen = groups.len(),
            stat = stat,
            anslen = group_answers.len(),
            guesslen = guessables.len(),
        );
        let best = best_time(
            MAX_TURNS,
            group_answers,
            guessables,
            if hard { Some((0, vec![])) } else { None },
            &mut cache,
        );
        if let Some((best_moves, _, best_word)) = best {
            movessum += if best_word == starter {
                assert_eq!(best_moves, 1);
                1
            } else {
                best_moves + u32::try_from(group_answers.len()).unwrap()
            };
            // Show guess sequence for each answer
            for ans in group_answers {
                if ans == &starter {
                    println!("{}", starter);
                    continue;
                }
                if &best_word == ans {
                    println!("{},{}", starter, ans);
                    continue;
                }
                let mut current_answers = group_answers;
                let mut guesses = MAX_TURNS;
                let mut restriction = if hard { Some((0, vec![])) } else { None };
                let mut grouped;
                print!("{},", starter);
                while current_answers.len() > 1 {
                    // don't need to filter guessables; cache doesnt care.
                    if let Some((_, _, new_best)) = best_time(
                        guesses,
                        current_answers,
                        guessables,
                        restriction.clone(),
                        &mut cache,
                    ) {
                        grouped = group_by_score(
                            new_best,
                            current_answers,
                            current_answers.len() - 1,
                        )
                        .unwrap();
                        current_answers = &grouped[&score(new_best, ans)];
                        if &new_best != ans {
                            print!("{},", new_best);
                        }
                        if let Some(r1) = restriction {
                            let (_, r2) =
                                filter_guesses(new_best, score(new_best, ans), &[]);
                            restriction = Some(combine_restriction(&r1, &r2));
                        }
                    } else {
                        unreachable!("{} should have been cached", ans);
                    }
                    guesses -= 1;
                }
                assert_eq!(&current_answers[0], ans);
                println!("{}", current_answers[0]);
            }
        }
        eprintln!(
            "finish {col} {ii:3}/{glen:3} {stat:3} {anslen:3} {guesslen:5} {movessum:4}/{lensum:4}={running:.3} {best:?} {elapsed:?}",
            col = stat_colour(starter, stat),
            ii = i + 1,
            glen = groups.len(),
            stat = stat,
            anslen = group_answers.len(),
            guesslen = guessables.len(),
            movessum = movessum,
            lensum = lensum,
            running = f64::from(movessum) / f64::from(lensum),
            best = best,
            elapsed = tstart.elapsed(),
        );
        println!(
            "{stat:3} {col} {anslen:3} {guesslen:5} {best:?} {group_answers:?}",
            stat = stat,
            col = stat_colour(starter, stat),
            anslen = group_answers.len(),
            best = best,
            guesslen = guessables.len(),
            group_answers = group_answers,
        );
        if best.is_none() {
            // If any group can't be solved, no point continuing with the word.
            break;
        }
    }

    if lensum == u32::try_from(answers.len()).unwrap() {
        println!(
            "{movessum:4}/{lensum:4}={running:.3}",
            movessum = movessum,
            lensum = lensum,
            running = f64::from(movessum) / f64::from(lensum),
        );
    }
}

fn main() {
    use std::env::args;
    use std::time::Instant;

    let answers: Vec<_> = include_str!("../dat/answers.txt").lines().collect();
    let words: Vec<_> = include_str!("../dat/wordlist.txt").lines().collect();

    if args().any(|a| a == "-f") {
        let mut hard = false;
        for arg in args().skip(1) {
            if arg == "-f" {
                continue;
            }
            if arg == "-h" {
                hard = true;
                continue;
            }
            if arg == "-e" {
                hard = false;
                continue;
            }
            fully_explore_starter(&arg, &answers, &words, hard);
        }
    } else {
        let mut nargs = args().len();
        let mut hard = false;
        if args().any(|a| a == "-h") {
            hard = true;
            nargs -= 1;
        }
        if nargs == 1 {
            // just to get the interesting results faster,
            // sort first.
            let sorted = group_by_score_and_sort(&words, &answers, answers.len());
            for (starter, _) in sorted {
                let v = rate_starter(starter, &answers, &words, hard);
                println!("{} {} {:?}", v.0, starter, v);
            }
        } else {
            for arg in args().skip(1) {
                if arg == "-h" {
                    continue;
                }
                let tstart = Instant::now();
                let v = rate_starter(&arg, &answers, &words, hard);
                println!("{} {} {:?} {:?}", v.0, arg, v, tstart.elapsed());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{best_time, combine_restriction, filter_guesses, score};
    use std::collections::HashMap;

    #[test]
    fn score_basic_yellow() {
        assert_eq!(score("abbbb", "caccc"), 1);
        assert_eq!(score("babbb", "acccc"), 3);
        assert_eq!(score("bbabb", "acccc"), 9);
        assert_eq!(score("bbbab", "acccc"), 27);
        assert_eq!(score("bbbba", "acccc"), 81);
    }

    #[test]
    fn score_basic_green() {
        assert_eq!(score("abbbb", "acccc"), 2);
        assert_eq!(score("babbb", "caccc"), 6);
        assert_eq!(score("bbabb", "ccacc"), 18);
        assert_eq!(score("bbbab", "cccac"), 54);
        assert_eq!(score("bbbba", "cccca"), 162);
    }

    #[test]
    fn score_multi_yellow() {
        assert_eq!(score("aabbb", "ccaac"), 4);
        assert_eq!(score("aabbb", "cccac"), 1);
        assert_eq!(score("aabbb", "ccaaa"), 4);
    }

    #[test]
    fn score_green_priority_over_yellow() {
        assert_eq!(score("aabbb", "caccc"), 6);
    }

    #[test]
    fn score_multi_green() {
        assert_eq!(score("aabbb", "aaccc"), 8);
        assert_eq!(score("aabbb", "aaacc"), 8);
        assert_eq!(score("aaabb", "aaccc"), 8);
    }

    #[test]
    fn score_green_plus_yellow() {
        assert_eq!(score("aabbb", "acacc"), 5);
        assert_eq!(score("baabb", "acacc"), 21);
    }

    #[test]
    fn combine_restriction_basic() {
        let r1 = (0b011, vec![('a', 1), ('b', 2)]);
        let r2 = (0b101, vec![('c', 1), ('b', 3)]);
        let r3 = (0b111, vec![('a', 1), ('b', 3), ('c', 1)]);
        assert_eq!(combine_restriction(&r1, &r2), r3);
    }

    #[test]
    fn filter_guesses_grey() {
        assert_eq!(
            filter_guesses("aaaaa", 0, &vec!["aaaaa", "bbbbb"]),
            (vec!["aaaaa", "bbbbb"], (0, vec![]))
        );
    }

    #[test]
    fn filter_guesses_basic_yellow() {
        let restrict = || (0, vec![('a', 1)]);

        assert_eq!(
            filter_guesses("abbbb", 1, &vec!["bbbbb"]),
            (Vec::<&str>::new(), restrict()),
        );

        let assert_allow = |a: &str, s: u8, b: &str| {
            assert_eq!(filter_guesses(a, s, &vec![b]), (vec![b], restrict()));
        };

        assert_allow("abbbb", 1, "acccc");
        assert_allow("abbbb", 1, "caccc");
        assert_allow("abbbb", 1, "ccacc");
        assert_allow("abbbb", 1, "cccac");
        assert_allow("abbbb", 1, "cccca");

        assert_allow("abbbb", 1, "acccc");
        assert_allow("babbb", 3, "acccc");
        assert_allow("bbabb", 9, "acccc");
        assert_allow("bbbab", 27, "acccc");
        assert_allow("bbbba", 81, "acccc");

        assert_allow("abbbb", 1, "aaccc");

        assert_allow("aabbb", 1, "ccacc");
    }

    #[test]
    fn filter_guesses_multi_yellow() {
        assert_eq!(
            filter_guesses("aabbb", 4, &vec!["abbbb"]),
            (Vec::<&str>::new(), (0, vec![('a', 2)]))
        );
        assert_eq!(
            filter_guesses("aabbb", 4, &vec!["ccaac"]),
            (vec!["ccaac"], (0, vec![('a', 2)]))
        );
    }

    #[test]
    fn filter_guesses_basic_green() {
        let restrict = |v: u8| (v, vec![('a', 1)]);

        assert_eq!(
            filter_guesses("abbbb", 2, &vec!["bbbbb"]),
            (Vec::<&str>::new(), restrict(1))
        );
        assert_eq!(
            filter_guesses("abbbb", 2, &vec!["babbb"]),
            (Vec::<&str>::new(), restrict(1))
        );

        let assert_allow = |a: &str, s: u8, b: &str, v: u8| {
            assert_eq!(filter_guesses(a, s, &vec![b]), (vec![b], restrict(v)));
        };

        assert_allow("abbbb", 2, "acccc", 1);
        assert_allow("babbb", 6, "caccc", 2);
        assert_allow("bbabb", 18, "ccacc", 4);
        assert_allow("bbbab", 54, "cccac", 8);
        assert_allow("bbbba", 162, "cccca", 16);
    }

    #[test]
    fn filter_guesses_multi_green() {
        let restrict = || (3, vec![('a', 2)]);

        assert_eq!(
            filter_guesses("aabbb", 8, &vec!["ababb"]),
            (Vec::<&str>::new(), restrict())
        );
        assert_eq!(
            filter_guesses("aabbb", 8, &vec!["baabb"]),
            (Vec::<&str>::new(), restrict())
        );
        assert_eq!(
            filter_guesses("aabbb", 8, &vec!["aaccc"]),
            (vec!["aaccc"], restrict())
        );
    }

    #[test]
    fn filter_guesses_green_plus_yellow() {
        let restrict = |v: u8| (v, vec![('a', 2)]);

        assert_eq!(
            filter_guesses("aabbb", 7, &vec!["acacc"]),
            (Vec::<&str>::new(), restrict(2))
        );
        assert_eq!(
            filter_guesses("aabbb", 7, &vec!["caacc"]),
            (vec!["caacc"], restrict(2))
        );

        assert_eq!(
            filter_guesses("aabbb", 5, &vec!["acacc"]),
            (vec!["acacc"], restrict(1))
        );
        assert_eq!(
            filter_guesses("aabbb", 5, &vec!["caacc"]),
            (Vec::<&str>::new(), restrict(1))
        );
    }

    #[test]
    fn best_time_basic_including_5_green() {
        let mut cache = HashMap::new();
        let t = best_time(
            6,
            &vec!["aaaaa", "bbbbb"],
            &vec!["aaaaa", "bbbbb"],
            None,
            &mut cache,
        );
        // 1 for aaaaa, 2 for bbbbb
        assert_eq!(t, Some((3, 2, "aaaaa")));
    }

    #[test]
    fn best_time_basic_just_under_limit() {
        let mut cache = HashMap::new();
        let t = best_time(
            2,
            &vec!["aaaaa", "bbbbb"],
            &vec!["aaaaa", "bbbbb"],
            None,
            &mut cache,
        );
        // 1 for aaaaa, 2 for bbbbb
        assert_eq!(t, Some((3, 2, "aaaaa")));
    }

    #[test]
    fn best_time_basic_just_over_limit() {
        let mut cache = HashMap::new();
        let t = best_time(
            1,
            &vec!["aaaaa", "bbbbb"],
            &vec!["aaaaa", "bbbbb"],
            None,
            &mut cache,
        );
        // 2 for each
        assert_eq!(t, None);
    }

    #[test]
    fn best_time_after_partition() {
        let mut cache = HashMap::new();
        let t = best_time(
            6,
            &vec!["bbbbb", "baaaa", "caaaa"],
            &vec!["bbbbb", "baaaa", "caaaa"],
            None,
            &mut cache,
        );
        // 1 for bbbbb
        // 2 for baaaa and caaaa by using bbbbb to distinguish them
        assert_eq!(t, Some((5, 2, "bbbbb")));
    }

    #[test]
    fn best_time_cant_partition_enough_time() {
        let mut cache = HashMap::new();
        let t = best_time(
            6,
            &vec!["baaaa", "caaaa", "daaaa", "eaaaa", "faaaa", "gaaaa"],
            &vec!["baaaa", "caaaa", "daaaa", "eaaaa", "faaaa", "gaaaa"],
            None,
            &mut cache,
        );
        assert_eq!(t, Some((21, 6, "baaaa")));
    }

    #[test]
    fn best_time_cant_partition_not_enough_time() {
        let mut cache = HashMap::new();
        let t = best_time(
            5,
            &vec!["baaaa", "caaaa", "daaaa", "eaaaa", "faaaa", "gaaaa"],
            &vec!["baaaa", "caaaa", "daaaa", "eaaaa", "faaaa", "gaaaa"],
            None,
            &mut cache,
        );
        assert_eq!(t, None);
    }
}
