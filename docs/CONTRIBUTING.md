# Contribution Guideline

## Using `issues` to communicate contributions

- Before working on a topic, please open a corresponding issue on GitHub and assign yourself.
- It is also reasonable to assign yourself to existing issues. 
- If you notice a bug, find a possible improvement, have a question, want to start a discussion, that you do not intend
to solve on your own, open an issue without assigning yourself. 
- Issues should describe the problem and provide helpful references if applicable. 
- If you are working on an issue, make sure to regularly update the issue with used references or partial solutions.
- Close the issue once the pull request, which solves the issue, is merged.

## Using `branches` to separate work from others

- Once you work on an issue, make sure to set up a branch for it. Please choose short and descriptive branch names.
- No work should be done on the `main` branch directly.
- Once your work with this branch is finished, open a pull-request to `main`.

## Commits 

- We support regular commits that show the incremental process of an issue.
- Commits should have a title that is a full sentence and shorter than 80 characters.
- Whenever applicable, additional commentary on commits is separated with a free line and a description on _why_ the
changes where undertaken. Please renounce on explaining _what_ happened, since it is most of the time obvious in the
diff.
- You do not need to clean your commit history, but please indicate incomplete commits with `WIP:`.

## Pull requests

- Pull requests are a robustness and safety measure for the main branch.
- Someone other than the requester should review the PR prior to the merge. 
- It is desired to leave comments, tasks and questions.
- Before merging, `git rebase` w.r.t. the main branch is desired. If you are not sure what this is or how to do it,
consult with the team.
- PRs should also verify that the code style roughly aligns with the project
