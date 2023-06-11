# honeyhive-take-home

We address the following:<br>
<br>
	1.	Provide a topic analysis on what kinds of inputs and outputs the prompt template fails on <br>
  &nbsp;We take the subset of the data where the e-mails are not accepted. We use Latent Dirichlet Allocation (LDA) [1] to extract the $n$ words most associated with $k$ topics. We then print the words for each topic for the failures of: inputs and outputs. Normally, one can use plots to determine how many words and topic to use. However, given the small dataset, we see from printing out words for topics that if we go above 5 words and 3 topics, either the words do not give us much intuition or the topics appear to be somewhat redundant.<br>
  <br>
	2.	Analyze the model outputs for problematic behaviors<br>
	&nbsp;We use tf-idf to construct features for each e-mail. We then fit logistic regression, regressing the logit of the response probability on the tf-idf features. When making predictions, we can look at the tf-idf features, and consider these words to be indicative of problematic behaviors that are _predictive_ of failure. Note that they may not in fact cause failure. We could use the critique, but it would be more difficult to steer a new response away from a topic than away from words in a vocabulary. The ideal would be to still avoid specific words, but use methods to ensure causal effects. This is more challenging and we consider this beyond the scope of the exercise.<br>
	<br>
	3.	Suggest improvements to the original prompt template<br>
	&nbsp;Our naive suggestion is to remove words in the e-mail that are predictive of failure, although this could lead to a non-sensical sentence. We coded this up, suggesting removing the two (a hyperparameter) words with the smallest effect on the linear preidctor. One could also re-generate the e-mail using a decoder language model and eliminate any predicted words that have large magnitude negative coefficients in the logistic regression model.<br>
	<br>
	4.	Suggest evaluation criteria to compare prompt templates<br>
	&nbsp;For two e-mails, we can pass both of them through the logistic regression model to predict their acceptance probability. We can consider the one with a higher probability of success to be the better e-mail.<br>

Our approach has two obvious limitations: first, the words that we propose either removing or not including do not necessarily have a strong _causal_ effect on acceptance probabilities, only predictive. Second, we are using handcrafted (tf-idf) features instead of word embeddings. For the first limitation, an extension could be to attempt to use topics extracted from the critiques and try to use those to steer the topics of the (revised) e-mail. However, this would be much more challenging. For the second limitation, we could use word embeddings instead of a pre-trained model, although because of the larger number of features when using word embeddings, one would likely either need a pre-trained model that we fine tune, a larger dataset, or dimensionality reduction to avoid overfitting.<br>

Further extension include using non-linearity instead of logistic regression with tf-idf features. That would likely improve predictive performance for whether an e-mail is accepted or not, but also make it more difficult to make simple recommendations such as 'delete this word from the vocabulary' though.<br>

[1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." Journal of machine Learning research 3.Jan (2003): 993-1022.
