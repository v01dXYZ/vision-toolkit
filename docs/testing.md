# Testing `vision-toolkit`

We test the library with:

- **unit** tests: verify the inner working of the library works as expected.
- **integration** tests: test cases with a given input and an expected output.
- **performance/regression** tests: based on datasets documented by papers, we compute a score for the results and how long it took to get them.

We prefer integration tests over unit tests as they are easier to understand and maintain.

## Test Datasets


| Dataset Name   | Description                                                                               | Link                                                                | labelled   | status  |
|----------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------|--------------| ---------|
| `Hollywood2`   | Multiple viewers watch different short scenes from movies. Coordinates are 2D cartesian.  | [paper](https://bop.unibe.ch/JEMR/article/view/JEMR.13.4.5/10052)   | yes |   OK (Segmentation)    |
| `Zemblys`      | Viewers look at a target moving on different points on a screen chosen randomly.) |                                                                                                                         | yes |   OK (Segmentation)   |
| `ETRA2019`     | 8 viewers look at scenes while performing a task. Both eyes are tracked. **IMPORTANT**: Data are not labelled. Only trajectories are available.                                             | [website](https://etra.acm.org/2019/challenge.html)                                                              | no | Maybe for performance regression |
