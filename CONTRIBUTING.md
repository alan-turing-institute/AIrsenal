# Contributing to AIrsenal

**Welcome to the AIrsenal repository!**
We're excited you're here and want to contribute.

We hope that these guidelines make it as easy as possible to get involved.
If you have any questions that aren't discussed below, please let us know by opening an [issue](https://github.com/alan-turing-institute/AIrsenal/issues).

We welcome all contributions from documentation to testing to writing code.
Don't let trying to be perfect get in the way of being good - exciting ideas are more important than perfect pull requests.

## Table of contents

- [Where to start: issues](#where-to-start-issues)
- [Making a change with a pull request](#making-a-change-with-a-pull-request)
  - [1. Comment on an existing issue or open a new issue referencing your addition](#1-comment-on-an-existing-issue-or-open-a-new-issue-referencing-your-addition)
  - [2. Fork the AIrsenal repository to your profile](#2-fork-the-airsenal-repository-to-your-profile)
  - [3. Make the changes you've discussed](#3-make-the-changes-youve-discussed)
  - [4. Submit a pull request](#4-submit-a-pull-request)
- [Style guide](#style-guide)

## Where to start: issues

* **Issues** are individual pieces of work that need to be completed to move the project forwards.
A general guideline: if you find yourself tempted to write a great big issue that
is difficult to describe as one unit of work, please consider splitting it into two or more issues.

Before you open a new issue, please check if any of our [open issues](https://github.com/alan-turing-institute/AIrsenal/issues) covers your idea already.

The list of labels for current issues includes:

- [![help-wanted](https://img.shields.io/badge/-help%20wanted-159818.svg)][labels-helpwanted] _These issues contain a task that a member of the team has determined we need additional help with._

  If you feel that you can contribute to one of these issues, we especially encourage you to do so!

- [![question](https://img.shields.io/badge/-question-cc317c.svg)][labels-question] _These issues contain a question that you'd like to have answered._

  Opening an issue is a great way to start a conversation and get your answer.

- [![good-first-issue](https://img.shields.io/badge/-good%20first%20issue-1b3487.svg)][labels-firstissue] _These issues are particularly appropriate if it is your first contribution to AIrsenal, or to GitHub overall._

- [![Enhancement](https://img.shields.io/badge/-enhancement-84b6eb.svg)][labels-enhancement] _These issues are suggesting new features that can be added to the project._

  If you want to ask for something new, please try to make sure that your request is distinct from any others that are already in the queue (or part of AIrsenal).
  If you find one that's similar but there are subtle differences please reference the other enhancement in your issue.

- [![Bug](https://img.shields.io/badge/-bug-d73a4a.svg)][labels-bug] _These issues are reporting a problem or a mistake in the project._

  The more details you can provide the better!
  If you know how to fix the bug, please open an issue first and then submit a pull request.

- [![project-management](https://img.shields.io/badge/-project%20management-bfd86c.svg)][labels-project-management] _We like to model best practice, so AIrsenal itself is managed through these issues.

## Making a change with a pull request

We appreciate all contributions to AIrsenal.
**THANK YOU** for helping us.

All project management, conversations and questions related to the AIrsenal project happens here in the [AIrsenal repository][AIrsenal-repo].

The following steps are a guide to help you contribute in a way that will be easy for everyone to review and accept with ease.

### 1. Comment on an [existing issue](https://github.com/alan-turing-institute/AIrsenal/issues) or open a new issue referencing your addition

This allows other members of the AIrsenal team to confirm that you aren't overlapping with work that's currently underway and that everyone is on the same page with the goal of the work you're going to carry out.

[This blog](https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/) is a nice explanation of why putting this work in up front is so useful to everyone involved.

### 2. [Fork][github-fork] the [AIrsenal repository][AIrsenal-repo] to your profile

This is now your own unique copy of AIrsenal.
Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date][github-syncfork] with the master repository, otherwise you can end up with lots of dreaded [merge conflicts][github-mergeconflicts].

### 3. Make the changes you've discussed

Try to keep the changes focused.
If you submit a large amount of work all in one go it will be much more work for whomever is reviewing your pull request.

While making your changes, commit often and write good, detailed commit messages.
[This blog](https://chris.beams.io/posts/git-commit/) explains how to write a good Git commit message and why it matters.
It is also perfectly fine to have a lot of commits - including ones that break code.
A good rule of thumb is to push up to GitHub when you _do_ have passing tests then the continuous integration (CI) has a good chance of passing everything.

If you feel tempted to "branch out" then please make a [new branch][github-branches] and a [new issue][AIrsenal-issues] to go with it. [This blog](https://nvie.com/posts/a-successful-git-branching-model/) details the different Git branching models.

Please do not re-write history!
That is, please do not use the [rebase](https://help.github.com/en/articles/about-git-rebase) command to edit previous commit messages, combine multiple commits into one, or delete or revert commits that are no longer necessary.

### 4. Submit a [pull request][github-pullrequest]

We encourage you to open a pull request as early in your contributing process as possible.
This allows everyone to see what is currently being worked on.
It also provides you, the contributor, feedback in real time from both the community and the continuous integration as you make commits (which will help prevent stuff from breaking).

When you are ready to submit a pull request, make sure the contents of the pull request body do the following:
- Describe the problem you're trying to fix in the pull request, reference any related issues and use keywords fixes/close to automatically close them, if pertinent.
- List changes proposed in the pull request.
- Describe what the reviewer should concentrate their feedback on.

If you have opened the pull request early and know that its contents are not ready for review or to be merged, add "[WIP]" at the start of the pull request title, which stands for "Work in Progress".
When you are happy with it and are happy for it to be merged into the main repository, change the "[WIP]" in the title of the pull request to "[Ready for review]".

A member of the AIrsenal team will then review your changes to confirm that they can be merged into the main repository.
A [review][github-review] will probably consist of a few questions to help clarify the work you've done.
Keep an eye on your GitHub notifications and be prepared to join in that conversation.

You can update your [fork][github-fork] of AIrsenal [repository][AIrsenal-repo] and the pull request will automatically update with those changes.
You don't need to submit a new pull request when you make a change in response to a review.

You can also submit pull requests to other contributors' branches!
Do you see an [open pull request](https://github.com/alan-turing-institute/AIrsenal/pulls) that you find interesting and want to contribute to?
Simply make your edits on their files and open a pull request to their branch!

What happens if the continuous integration (CI) fails (for example, if the pull request notifies you that "Some checks were not successful")?
The CI could fail for a number of reasons.
At the bottom of the pull request, where it says whether your build passed or failed, you can click “Details” next to the test, which takes you to the Travis page.
You can view the log or rerun the checks if you have write access to the repo by clicking the “Restart build” button in the top right (you must be logged in to Travis CI with your GitHub account see the “Restart build” button).

GitHub has a [nice introduction][github-flow] to the pull request workflow, but please get in touch if you have any questions.

## Coding Conventions and Style Guide

We have attempted to summarize some guidelines and helpful tips for keeping the code readable and consistent in [this document](CodingConventions.md).

---

_These Contributing Guidelines have been adapted from the [Contributing Guidelines](https://github.com/bids-standard/bids-starter-kit/blob/master/CONTRIBUTING.md) of [The Turing Way](https://github.com/alan-turing-institute/the-turing-way)! (License: MIT)_

[AIrsenal-repo]: https://github.com/alan-turing-institute/AIrsenal/
[AIrsenal-issues]: https://github.com/alan-turing-institute/AIrsenal/issues
[git]: https://git-scm.com
[github]: https://github.com
[github-branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository
[github-fork]: https://help.github.com/articles/fork-a-repo
[github-flow]: https://guides.github.com/introduction/flow
[github-mergeconflicts]: https://help.github.com/articles/about-merge-conflicts
[github-pullrequest]: https://help.github.com/articles/creating-a-pull-request
[github-review]: https://help.github.com/articles/about-pull-request-reviews
[github-syncfork]: https://help.github.com/articles/syncing-a-fork
[labels-bug]: https://github.com/alan-turing-institute/AIrsenal/labels/bug
[labels-enhancement]: https://github.com/alan-turing-institute/AIrsenal/labels/enhancement
[labels-firstissue]: https://github.com/alan-turing-institute/AIrsenal/labels/good%20first%20issue
[labels-helpwanted]: https://github.com/alan-turing-institute/AIrsenal/labels/help%20wanted
[labels-project-management]: https://github.com/alan-turing-institute/AIrsenal/labels/project%20management
[labels-question]: https://github.com/alan-turing-institute/AIrsenal/labels/question
