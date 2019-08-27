.. highlight:: none

==============
 Contributing
==============

All contributions are welcome, whether they be usage cases, bug
reports, or code contributed through pull requests. In the latter
case, we ask that you follow the code style, testing, and version
control guidelines outlined in this document to make it as seamless as
possible to integrate your changes into the repository!

Code Style
==========

Code itself should be formatted with Black_ on default settings. This
gives a consistent and readable style to all Python files in the
codebase. You can run Black manually on changed files, or `integrate
it into git`__.

.. _Black: https://github.com/psf/black
.. __: https://black.readthedocs.io/en/stable/version_control_integration.html

Any public methods and classes that will show up on the :doc:`api`
page should have a docstring describing its function, arguments, and
return values. The docstrings are automatically turned into
documentation by the Napoleon_ sphinx extension, which expects
formatting to follow the `Google Python Style Guide`__.

.. _Napoleon: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
.. __: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

Testing
=======

All pull requests are tested against a suite of unit and integration
tests. For fixes or changes to existing functionality, the test suite
should continue passing, unless of course a test was based on the code
that was changed. In this case, and with new functionality, tests
should be modified and added as required.

Tests are logically grouped in the ``test/`` directory. Groups of unit
tests should be named accordingly, whereas most large-scale
integration tests will probably end up in ``test_filtering.py``.

To run the tests locally, install the required dependencies with

.. code-block:: console

   $ pip install -e '.[build]'

or simply install `pytest` manually. From the root directory of the
project, running ``pytest`` will execute the entire test suite.

Version Control
===============

`git` is a powerful tool, but it's very easy to make things
messy. Commits should be used for a logical, contained change to the
code. They should be formatted as follows::

  Short (72 chars or less) summary of changes

  Explanatory text, wrapped to 72 characters. Make sure this is
  separated from the summary by a blank line.

  More text, if you need it. If your commit is targeting a particular
  issue, you can end the message with a line telling GitHub to
  automatically close it, like this:

  Closes #42

The summary line should be written in the imperative form, without a
terminating full stop. You should be able to insert your summary in
the following sentence: `This commit will <summary>`. Context
motivating the change, or additional details are welcomed in commit
messages so that they may stand on their own.

Pull requests should have a clean, linear sequence of commits. If you
have opened a pull request, you own the history of that branch. This
means that you are free to rewrite history as required. This is useful
in particular in two cases:

1. Changes are made to ``master`` upon which your pull request should
   be based.

   In this case, you should ``git rebase`` to "replay" your commits
   onto the base branch, rather than introducing a merge commit.

2. You need to make a minor change to an existing commit on your
   branch.

   This happens often, if you're trying to appease the code formatter,
   or a typo/tweak is pointed out to you. Rather than adding a new
   commit (usually with a message like "fix typo"), feel free to
   rewrite history to incorporate the change into the existing
   commit. This can be done with an interactive rebase, or fixup
   commit that can be squashed in at a later time. See the relevant
   section of the `Git Book`_ for some inspiration.

Whenever you have rewritten history, you are likely to need a force
push. This is okay for pull requests and strictly personal branches,
but nowhere else!

.. _Git Book: https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History
