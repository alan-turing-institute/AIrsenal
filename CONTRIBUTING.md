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
- [Coding conventions](#coding-conventions)

## Where to start: issues

**Issues** are individual pieces of work that need to be completed to move the project forwards.
If you find yourself tempted to write a great big issue that is difficult to describe as one unit of work, please consider splitting it into two or more.

Before you open a new issue, please check whether one of our [open issues](https://github.com/alan-turing-institute/AIrsenal/issues) covers your idea already.

We use these labels:

- [help wanted][labels-helpwanted] - a task we'd particularly welcome help with.
- [good first issue][labels-firstissue] - a good place to start if it's your first contribution to AIrsenal, or to GitHub.
- [question][labels-question] - something you'd like answered. Opening one is a great way to start a conversation.
- [enhancement][labels-enhancement] - a suggested new feature. Please check it's distinct from anything already in the queue, and reference any similar issue you find.
- [bug][labels-bug] - a problem or mistake in the project. The more detail the better; if you know the fix, open the issue first and then a pull request.
- [project management][labels-project-management] - _we like to model best practice, so AIrsenal itself is managed through these issues._

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

Make sure to [keep your fork up to date][github-syncfork] with the main repository, otherwise you can end up with lots of dreaded [merge conflicts][github-mergeconflicts].

### 3. Make the changes you've discussed

Branch off `develop` and name your branch `feature/<issue-number>-<description>` or `bugfix/<issue-number>-<description>`.

Try to keep the changes focused.
If you submit a large amount of work all in one go it will be much more work for whomever is reviewing your pull request.

While making your changes, commit often and write good, detailed commit messages.
[This blog](https://chris.beams.io/posts/git-commit/) explains how to write a good Git commit message and why it matters.
It is also perfectly fine to have a lot of commits - including ones that break code.
A good rule of thumb is to push up to GitHub when you _do_ have passing tests then the continuous integration (CI) has a good chance of passing everything.

If you feel tempted to "branch out" then please make a [new branch][github-branches] and a [new issue][AIrsenal-issues] to go with it.

Please do not re-write history!
That is, please do not use the [rebase](https://help.github.com/en/articles/about-git-rebase) command to edit previous commit messages, combine multiple commits into one, or delete or revert commits that are no longer necessary.

### 4. Submit a [pull request][github-pullrequest]

Open your pull request against `develop`, and open it as early in your contributing process as possible.
This allows everyone to see what is currently being worked on.
It also provides you, the contributor, feedback in real time from both the community and the continuous integration as you make commits (which will help prevent stuff from breaking).

When you are ready to submit a pull request, make sure the contents of the pull request body do the following:
- Describe the problem you're trying to fix in the pull request, reference any related issues and use keywords fixes/close to automatically close them, if pertinent.
- List changes proposed in the pull request.
- Describe what the reviewer should concentrate their feedback on.

If it isn't ready for review yet, open it as a [draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request) and mark it ready for review when it is.

A member of the AIrsenal team will then review your changes to confirm that they can be merged into the main repository.
A [review][github-review] will probably consist of a few questions to help clarify the work you've done.
Keep an eye on your GitHub notifications and be prepared to join in that conversation.

You can update your [fork][github-fork] of AIrsenal [repository][AIrsenal-repo] and the pull request will automatically update with those changes.
You don't need to submit a new pull request when you make a change in response to a review.

You can also submit pull requests to other contributors' branches!
Do you see an [open pull request](https://github.com/alan-turing-institute/AIrsenal/pulls) that you find interesting and want to contribute to?
Simply make your edits on their files and open a pull request to their branch!

CI runs on GitHub Actions. If a check fails, click "Details" next to it on the pull request to see the log; pushing a new commit reruns everything. Most failures are the pre-commit hooks, which you can run locally first with `pre-commit run --all-files`.

GitHub has a [nice introduction][github-flow] to the pull request workflow, but please get in touch if you have any questions.

## Coding conventions

Guidelines for keeping the code readable and consistent are in [CodingConventions.md](CodingConventions.md).

---

_These Contributing Guidelines have been adapted from the [Contributing Guidelines](https://github.com/bids-standard/bids-starter-kit/blob/master/CONTRIBUTING.md) of [The Turing Way](https://github.com/alan-turing-institute/the-turing-way)! (License: MIT)_

[AIrsenal-repo]: https://github.com/alan-turing-institute/AIrsenal/
[AIrsenal-issues]: https://github.com/alan-turing-institute/AIrsenal/issues
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
