# Example guidelines

This document provides general guidelines when writing examples that will be integrated into the Connect documentation. It applies to both Substra and Substrafl examples. It is mainly based on the [Google developer documentation style guide](https://developers.google.com/style).

Please also read the `template.py` file, as it goes through the structure of an example.

In addition to that, you will find below more general guidelines:

- Use short sentences.
- Use clear and direct language. Avoid the use of double negatives.
- Have a friendly and conversational tone.
- Use standard American spelling.
- Spell check by a native speaker is to be done regularly and organized by the product manager responsible for the documentation. It does not have to be done at every PR.
- In general, use present tense rather than future tense; in particular, try to avoid using will where possible—for example:
  - Recommended: Send a query to the service. The server sends an acknowledgment.
  - Not recommended: Send a query to the service. The server will send an acknowledgment.
- In addition to being mindful of the pronouns used in narrative examples, be sensitive to other possible sources of gendered language.
- In general, use second person in your documents rather than first person—you instead of we. It's also important to figure out who the you is that you're addressing (a developer? a sysadmin? someone else?) and to be consistent about that.
  - Recommended: If you're deleting multiple entries at a time ...
  - Not recommended: If we're deleting multiple entries at a time ...
- In general, use active voice. In passive voice, it's easy to neglect to indicate who or what is performing a particular action. In this kind of construction, it's often hard for readers to figure out who's supposed to do something (such as the reader, the computer, the server, an end user, or a visitor to a web page).
  - Recommended: Send a query to the service. The server sends an acknowledgment.
  - Not recommended: The service is queried, and an acknowledgment is sent.
- Don't pre-announce anything in documentation: avoid trying to document future features or products, even in innocuous ways.

In particular for examples:

- Examples should be self-contained and should not use results or data from other examples.
- Examples should run in less than one minute in subprocess mode.
- Provide one image per example, so that the image is shown in the sphinx gallery.
- The example should come with its own directory with:
  - an "assets" directory containing the assets: data, algo, metric... and the requirements.txt
  - a file `plot_example_name.py`. This file will generate the equivalent of a computed notebook and mainly uses `rst` syntax.
  - an image used for the sphinx gallery
