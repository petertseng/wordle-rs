# WordleRS

Rust tools for Wordle.

Note (for those unfamiliar with Rust) that .rs is the standard extension for Rust files, thus WordleRS.

The aim is to minimise the mean number of guesses subject to the constraint of not failing to guess any word.
This specifically uses the 2315 possible answers and the 10657+2315 = 12972 valid guesses.
This because if all 12972 valid guesses were considered possible answers, not only would the analysis take longer, but it would also unfairly overvalue the letter S in the fifth position.

The analysis proceeds in two phases:

1. Approximation phase: For each word in the word list, the program outputs an approximation of how well it might perform as a first guess by:
    * Building the decision tree rooted at that first guess, where at each branch you choose only the one word that minimises the mean number of words in a bucket.
    * Counting the total number of guesses it takes to guess all 2315 possible answers.
    * Please understand that this is only an approximation. Minimising the mean number of words in a bucket doesn't minimise the mean number of guesses.
1. Optimisation phase: A human manually picks certain words (typically the N words that scored the best in the approximation phase, but other words may be manually chosen if desired) and instructs the program to perform a full analysis of how those words in fact perform as a first guess.
    * The full analysis considers every word at every possible branch so as to be sure not to miss a better possibility. Therefore, it finds the optimal decision tree.
    * Prune the search space: If a word being considered cannot beat the current best, it need not be considered.
    * Cache: If a set of words has already been seen before, the answer will be the same.
    * The above two points are needed to get this to run in reasonable time, but if they are implemented incorrectly they will also cause the code to potentially miss good solutions. So when reviewing the code it's important to make sure they are implemented correctly and/or test the code with and without the above two points on a smaller word list to see if it reaches the same answers.
    * Please be patient. The optimisation phase can take wildly varying amounts of time depending on the starting word. For example, for the words TRACE and CRATE (as well as many others) the bucket with yellow E and yellow R takes more than **30 minutes**.

## Hard mode

Hard mode is easier to approximate and harder to optimise.

It's easier to approximate simply because there are fewer valid guesses that you need to evaluate.

It's harder to optimise because the lack of some valid guesses means some buckets don't partition as cleanly, which invalidates some of the pruning that is possible in easy mode.
Further exploration of hard mode may reveal some pruning rule that still works effectively in hard mode.

## Results

The best word for easy mode is SALET, taking 7920 guesses in total (7920/2315=3.421 average).
All words are guessed in no more than five guesses.

A separate exhaustive search also determines that there is no decision tree that can find all 2315 words in no more than four guesses.

The best word for hard mode is also SALET, taking 8122 guesses in total (8122/2315=3.508 average).
All words are guessed in no more than six guesses.
