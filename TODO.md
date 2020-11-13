# TODO

## General
- [x] Rename/relocate project
- [ ] Add validation to all class members and throw meaningful warnings
  - [x] Add validation to configs
- [ ] Add enums where applicable

## Model
- [x] Create scheme class
- [x] Add custom metric function support
- [x] Element wise neighbours / adjencency matrix
- [x] Write states optionally to disk + config
- [x] Implement state memory
- [x] Create built-in conditions
- [x] Memory/output config class
  - [x] State memory
  - [x] Utility memory
  - [x] Adjacency memory
- [ ] Add non-array support
- [ ] Multi process update functions per iteration

## Dynamic network utility layer


### Edge values
- [x] Utility per edge
- [x] Utility change
  - [x] NxN matrix
  - [x] Specific edges
- [x] Optional utility initialization 
- [x] Threshold state conditions
- [x] Threshold utility
- [x] Threshold adjacency (amount of neighbors)
- [ ] Rework to edge values dictionary

### Utility
- [ ] Implement utility maximisation
- [ ] Implement cost function(s)

### Network updating
<!-- - [ ] Order update dictionary -> First utility then network? -->
- [x] Remove nodes
  - [x] List format
- [x] Edge changes
  - [x] Overwrite
  - [x] Add
  - [x] Remove
  - [x] Optional utility init
  - [ ] New adjacency matrix
    - [ ] New utility matrix for init
- [x] Add nodes
  - [x] Optional state init
  - [x] Optional edges 
    - [x] Optional utility init
- [] Optional utility callable initialization

## UI
- [ ] Add graphical UI for state / constant selection / distribution specification

## Update
- [x] Add support for conditions
- [x] Set all chained condition state indices

## Conditions
- [x] Custom condition

## Examples
- [x] Add example runner
- [ ] Add example param specification

## Sensitivity Analysis
- [x] Add sensitivity analysis runner
- [x] Add SA metrics
  - [x] Means
  - [x] Variance
  - [x] Min / Max
  - [x] Network metrics
  - [x] Custom function
- [ ] Rework to take model refactoring into account
- [ ] Add parallel processing

## Visualization
- [x] Add more layout and networkx layout support
- [x] Deal with different dimensions when writing
- [x] Reconfiguration visualization
- [x] More consistent network update locations
- [x] Read states from disk
  - [ ] Rework for dictionary format
- [x] Visualize utility
- [ ] Add regular plots / trends
- [ ] Optimize animation if possible
- [ ] Support jupyter notebook

## Testing
- [ ] Write more tests

## Documentation
- [x] Implement ReadTheDocs
- [ ] Document code

## DevOps
- [x] Add CI support
- [x] Add code quality checking
- [x] Add automatic coverage checking
- [ ] Create contribution and pull request templates

## Package
- [x] Publish to PyPi
- [x] Add license