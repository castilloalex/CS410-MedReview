## Cleaning the data

Originally there are 215,063 reviews. 

The final dataset contains **182,475 reviews across 3,540 distinct drug names**:

| Dataset | Before cleaning | After cleaning |
| --- | ---: | ---: |
| Training | 161,297 | 128,722 |
| Test | 53,766 | 53,753 |
| Total | 215,063 | 182,475 |

## Building the search index

The index works like a keyword directory: it records which reviews contain each word and how often that word appears. It is stored in a local SQLite database and contains 54,794 distinct terms. 

A basic BM25 search is already working. For example, searching for “sleep groggy” and filtering by “Trazodone” returns matching reviews ranked by relevance. Each result includes the review text, drug name, rating, and other details. Drug names must match, ignoring case; brand and generic names are not automatically linked.

The default search index includes both cleaned datasets. For sentiment training, the team should use only the training set and keep the test set for evaluation. A separate search index containing only training reviews can also be built.

## Checking the Results and Next Steps

All three automated tests and all 15 full-data checks passed. The tests use small examples to check cleaning, duplicate removal, search scores, and drug filters. The full-data checks confirm that the index matches the cleaned reviews and that no exact text overlap remains between training and test data under our rules. Detailed results are saved in `part1_results.json`.