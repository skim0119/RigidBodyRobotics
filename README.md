<div align='center'>
<h1> PyElastica-Rigid Simulation </h1>
 </div>

Experimental plugin of Rigid body motion within PyElastica framework.
This is just testing repo for utilizing elastica physics simulation framework.

> Use git-stack workflow.

## How to run

```sh
pip install "."
python examples/project1/task1.py
```

## How to pytest

```sh
pytest
```

## Dev status

- [x] Migrate PyElastica structure for rigid body and explicit-euler stepping
    - [x] Modify stepper to explicit-euler
    - [x] Stepper to symplectic-euler
- [x] Implement robot and its properties
- [x] Setup problem statement
    - [x] Add Roomba as part of the pyelastica-simulator system
    - [x] Implement environment: robot, field
    - [x] Friction on surface
    - [x] Wall collision
- [x] Equation
    - [x] SO2 addition
- [x] Examples
    - [x] task 1
    - [x] task 2
    - [x] task 3

## Notes

## References

The repository is originated to practice ME498 project.
