# Testing `vision-toolkit`

We test the library with:

- **unit** tests: verify the inner working of the library works as expected.
- **integration** tests: test cases with a given input and an expected output.
- **performance/regression** tests: based on datasets documented by papers, we compute a score for the results and how long it took to get them.

We prefer integration tests over unit tests as they are easier to understand and maintain.

## Test Datasets

### Segmentation


| Dataset Name   | Description                                                                               | Link                                                                | status  |
|----------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------|---------|
| `Hollywood2`   | Multiple viewers watch different short scenes from movies. Coordinates are 2D cartesian.  | [paper](https://bop.unibe.ch/JEMR/article/view/JEMR.13.4.5/10052)   |   OK    |
| `Zemblys`      | Viewers look at a target moving on different points on a screen 5chosen randomly per viewer.) |                                                                 |   OK    |
| `ETRA2019`     |                                                                                           |                                                                     | Planned |