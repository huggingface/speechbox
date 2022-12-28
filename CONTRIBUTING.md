<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# How to contribute to speechbox?

Everyone is welcome to contribute, and we value everybody's contribution. Code
is thus not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/speechbox/blob/main/CODE_OF_CONDUCT.md).

## Philosophy

`speechbox` is a easy-to-use, and easy-to-contribute package for all kinds of tasks related to *speech*.

We aim for the project to be extremely easy to use and extremely easy to contribute to.
This is achieved by enforcing the following structure:

* 1.) `speechbox` is simply a collection of tasks. Every task can be defined in **one and only one** single file, such as [restore.py](https://github.com/huggingface/speechbox/blob/main/src/speechbox/restore.py). Therefore, **no** abstraction between files is used. We prefer copy-pasting code at the expense of duplicated code. See [this blog post for some reasons behind it](https://huggingface.co/blog/transformers-design-philosophy)
* 2.) Every task that is added is maintained by the author/contributor of this task which can be find in [this table](https://github.com/huggingface/speechbox#tasks). Therefore, please make sure to tag the author/contributor if you find an issue with the task
* 3.) When adding a task, make sure to extend the [Task table](https://github.com/huggingface/speechbox#tasks), to add an example, such as [examples/restore.py](https://github.com/huggingface/speechbox/blob/main/examples/restore.py), and as stated in 1.) a single task file.
* 4.) We don't want to add any new **required dependencies** to [setup.py](https://github.com/huggingface/speechbox/blob/9160d26c0fcada3df54824bfaa511a4b7992e16d/setup.py#L156). Nevertheless, we stongly recommend making use of existing packages, but please make sure to add them as **soft-dependencies** by adding a `is_{your_package}_available` function [here](https://github.com/huggingface/speechbox/blob/main/src/speechbox/utils/import_utils.py) and only importing your tool when the library is available [in the public __init__.py](https://github.com/huggingface/speechbox/blob/main/src/speechbox/__init__.py).

## You can contribute in so many ways!

There are 4 ways you can contribute to speechbox:

**Contributions to get started**:

* Fixing outstanding issues with the existing code;
* Submitting issues related to bugs or desired new features.

In particular there is a special [Good First Issue](https://github.com/huggingface/speechbox/contribute) listing. 
It will give you a list of open Issues that are open to anybody to work on. Just comment in the issue that you'd like to work on it. 

**Advanced Contributions**:

For an advanced contribution, you can try to add a new task according to the philosophy as defined [above](#philosophy).

You can check open feature requests for tasks/tools under [New Task](https://github.com/huggingface/speechbox/labels/New%20Task) listing. 

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

### Do you want a new task ?

A world-class feature request issue addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain
    why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear
    about it!
  * Is it something you worked on and think could benefit the community?
    Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

Please make sure to add `[New task]` to the title of your issue.

## Start contributing! (Pull Requests)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
`speechbox`. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing.

1. Fork the [repository](https://github.com/huggingface/speechbox) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/speechbox.git
   $ cd speechbox
   $ git remote add upstream https://github.com/huggingface/speechbox.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   $ pip install -e ".[dev]"
   ```

   (If speechbox was already installed in the virtual environment, remove
   it with `pip uninstall speechbox` before reinstalling it in editable
   mode with the `-e` flag.)

5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this:

   ```bash
   $ pytest tests/<TEST_TO_RUN>.py
   ```

   `speechbox` also uses `flake8` and a few custom scripts to check for coding mistakes. Quality
   control runs in CI, however you can also run the same checks with:

   ```bash
   $ make quality
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`. These
   are useful to avoid duplicated work, and to differentiate it from PRs ready
   to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests. No quality testing = no merge.
   - If you are adding new `@slow` tests, make sure they pass using
     `RUN_SLOW=1 python -m pytest tests/test_my_new_model.py`.
   - If you are adding a new tokenizer, write tests, and make sure
     `RUN_SLOW=1 python -m pytest tests/test_tokenization_{your_model_name}.py` passes.
   CircleCI does not run the slow tests, but github actions does every night!
6. All public methods must have informative docstrings that work nicely with sphinx. See `modeling_bert.py` for an
   example.
7. Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
   the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference 
   them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
   If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
   to this dataset.

### Style guide

For documentation strings, `speechbox` follows the [google style](https://google.github.io/styleguide/pyguide.html).

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

### Syncing forked main with upstream (HuggingFace) main

To avoid pinging the upstream repository which adds reference notes to each upstream PR and sends unnecessary notifications to the developers involved in these PRs,
when syncing the main branch of a forked repository, please, follow these steps:
1. When possible, avoid syncing with the upstream using a branch and PR on the forked repository. Instead merge directly into the forked main.
2. If a PR is absolutely necessary, use the following steps after checking out your branch:
```
$ git checkout -b your-branch-for-syncing
$ git pull --squash --no-commit upstream main
$ git commit -m '<your message without GitHub references>'
$ git push --set-upstream origin your-branch-for-syncing
```
