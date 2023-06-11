# honeyhive-take-home

We address the following:<br>
	1.	Provide a topic analysis on what kinds of inputs and outputs the prompt template fails on <br>
  &nbsp;We take the subset of the data where the e-mails are not accepted. We use Latent Dirichlet Allocation (LDA) [1] to extract the $n$ words most associated with $k$ topics. We then print the words for each topic for the failures of: inputs and outputs.<br>
	2.	Analyze the model outputs for problematic behaviors<br>
	3.	Suggest improvements to the original prompt template<br>
	4.	Suggest evaluation criteria to compare prompt templates<br>

[1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." Journal of machine Learning research 3.Jan (2003): 993-1022.
