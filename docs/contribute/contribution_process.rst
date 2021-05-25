Getting Started
===============

Here is the basic process.

-  **Figure out what you're going to work on.** The majority of the
   contributions come from people scratching their own itches.
   However, if you don't know what you want to work on, or are just
   looking to get more acquainted with the project, look through the `issue
   tracker <https://github.com/owkin/connectlib/issues/>`__ and see if
   there are any issues you know how to fix. Issues that are
   confirmed by other contributors tend to be better to investigate.

-  **Figure out the scope of your change and reach out for design
   comments on a GitLab issue if it's large.** The majority of pull
   requests are small; in that case, no need to let us know about what
   you want to do, just get cracking. But if the change is going to be
   large, it's usually a good idea to get some design comments about it
   first.

   -  If you don't know how big a change is going to be, we can help you
      figure it out! Just post about it on issues.
   -  Some feature additions are very standardized; for example, lots of
      people add strategies to ConnectLib. Design
      discussion in these cases boils down mostly to, “Do we want this
      module/algorithm?”
   -  Core changes and refactors can be quite difficult to coordinate,
      as the pace of development on ConnectLib master is quite fast.
      Definitely reach out about fundamental or cross-cutting changes;
      we can often give guidance about how to stage such changes into
      more easily reviewable pieces.

-  **Code it out!**

   -  See the `coding conventions <technical_guide.html>`__ for advice for working with ConnectLib in a
      technical form.

-  **Open a pull request.**

   -  If you are not ready for the pull request to be reviewed, tag it
      with [WIP]. We will ignore it when doing review passes. If you are
      working on a complex change, it's good to start things off as WIP,
      because you will need to spend time looking at CI results to see
      if things worked out or not.
   -  Find an appropriate reviewer for your change.

-  **Iterate on the pull request until it's accepted!**

   -  We'll try our best to minimize the number of review roundtrips and
      block PRs only when there are major issues. For the most common
      issues in pull requests, take a look at `Common Mistakes <mistakes.html>`__.
   -  Once a pull request is accepted and CI is passing, there is
      nothing else you need to do; we will merge the PR for you.


------------------------

Ressources
----------

Proposing new features
~~~~~~~~~~~~~~~~~~~~~~

New feature ideas are best discussed on a specific issue. Please include
as much information as you can, any accompanying data, and your proposed
solution. If you feel confident in your solution, go ahead and implement it.

Reporting Issues
~~~~~~~~~~~~~~~~

If you've identified an issue, first search through the `list of
existing issues <https://github.com/owkin/connectlib/issues>`__ on the
repo. If you are unable to find a similar issue, then create a new one.
Supply as much information you can to reproduce the problematic
behavior. Also, include any additional insights like the behavior you
expect.

Implementing Features or Fixing Bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to fix a specific issue, it's best to comment on the
individual issue with your intent.
It's best to strike up a conversation on the issue and discuss your
proposed solution. Maintainers can provide guidance that saves you
time.

Improving Documentation & Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We aim to produce high quality documentation and tutorials. On rare
occasions that content includes typos or bugs. If you find something you
can fix, send us a pull request for consideration.

Submitting pull requests to fix open issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can view a list of all open issues
`here <https://github.com/owkin/connectlib/issues>`__. Commenting on an
issue is a great way to get the attention of the maintainers. From here you can
share your ideas and how you plan to resolve the issue.

For more challenging issues, the team will provide feedback and
direction for how to best solve the issue.

If you're not able to fix the issue itself, commenting and sharing
whether you can reproduce the issue can be useful for helping the team
identify problem areas.

Reviewing open pull requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We appreciate your help reviewing and commenting on pull requests. Our
team strives to keep the number of open pull requests at a manageable
size, we respond quickly for more information if we need it, and we
merge PRs that we think are useful. However, due to the high level of
interest, additional eyes on pull requests is appreciated.

Improving code readability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Improve code readability helps everyone. It is often better to submit a
small number of pull requests that touch few files versus a large pull
request that touches many files. Starting an issue related to your improvement
is the best way to get started.

Adding test cases to make the codebase more robust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional test coverage is appreciated.
