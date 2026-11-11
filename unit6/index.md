---
layout: notes
title: "Unit 6: Regularization, Priors & Bayesian Inference"
unit_title: "Unit 6"
unit_subtitle: "Overparameterized models: priors, posteriors, and regularization"
toc:
  - {href: "#sec-6-1", label: "6.1 Bayesian inference: simple examples"}
  - {href: "#sec-6-2", label: "6.2 Bayesian linear regression"}
  - {href: "#sec-6-3", label: "6.3 Regularization view of priors"}
  - {href: "#sec-6-4", label: "6.4 The kernel trick", optional: true}
  - {href: "#problems", label: "Problems"}
---

<div class="unit-overview" markdown="1">

We've seen that models can be too complicated, but so far **complexity** has roughly meant the number of parameters (regression coefficients), and balancing bias and variance has meant choosing the number of predictors to include. There are also ways to build very flexible models that don't overfit. This unit introduces **Bayesian inference** as an alternative to the frequentist framework we've used so far, and shows how it leads naturally to **regularization**, which penalizes model flexibility directly. With regularization we can include many predictors (even more than data points) without the sample distribution blowing up.

#### Concepts

Frequentist vs. Bayesian treatment of parameters, priors and posteriors, Bayes' theorem, the Beta and Normal conjugate models, Laplace's rule of succession, Bayesian linear regression, the posterior mean for regression coefficients, posterior prediction, ridge regression, the connection between regularization and priors, and (optional) the kernel trick.

#### Things to practice

- Select and justify priors (using Normal or log-normal probabilities).
- For estimating the sample mean, calculate the influence of a given regularization on the least-squares estimator by hand.
- Implement regularization in Python and have a back-of-the-envelope sense of its effect on regression coefficients.
- Translate priors into regularization for ridge regression, and vice versa.

</div>

<p class="pdf-link"><a href="unit6.pdf">Unit 6 notes pdf</a> <a href="https://colab.research.google.com/github/elevien/math50_2026/blob/main/unit6/unit6.ipynb">Unit 6 notebook</a></p>

## 6.1 Bayesian inference: simple examples {#sec-6-1}

So far, our models have encoded assumptions about how data was generated, depending on parameters whose true values are fixed. When we use data to <span class="term">[fit](https://en.wikipedia.org/wiki/Estimation_theory)</span> or <span class="term">[infer](https://en.wikipedia.org/wiki/Statistical_inference)</span> parameters, we obtain a sample distribution, which roughly speaking quantifies our uncertainty. This way of doing statistics &mdash; where uncertainty is a distribution over repeated experiments &mdash; is the <span class="term">[frequentist](https://en.wikipedia.org/wiki/Frequentist_inference)</span> approach. In this formulation we pretend to have complete ignorance of the parameters before seeing the data, but in reality that's never quite true. For example, if we flip a coin $N=3$ times and see $Y=3$ heads, our estimate of the probability of heads is $\hat q=1$ &mdash; clearly not in line with our understanding that both outcomes are at least possible. Worse, the usual standard error $\text{se}(\hat q)=\sqrt{q(1-q)/N}\approx\sqrt{\hat q(1-\hat q)/N}$ comes out to zero, which is obviously not a reasonable estimate of our uncertainty.

Similar issues come up in more machine-learning-style analysis: sometimes we want to incorporate vague prior information, like "the function describing my data is smooth and changes on a time-scale of about 5 hours." Or we might include many predictors but want to penalize large coefficients to avoid overfitting &mdash; e.g. we might believe an interaction term exists but suspect it's much smaller than the additive terms.

There are ways to handle these within the frequentist framework, but they're handled more naturally by a different treatment of parameters altogether: <span class="term">[Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics)</span>. Here we treat parameters themselves as random variables, given a distribution *before* we see the data. Instead of a model with a fixed parameter,

$$ X \sim \text{ModelDistribution}(\theta), $$

we condition on $\theta$:

$$ X\mid\theta \sim \text{ModelDistribution}(\theta). $$

We call $X\mid\theta$ the <span class="term">[likelihood](https://en.wikipedia.org/wiki/Likelihood_function)</span> (a term used in both the frequentist and Bayesian approaches). It is the distribution of the observed data given a specific value of the unknown parameter $\theta$, so it tells us how likely the observed data are under different values of $\theta$. In Bayesian inference we also give $\theta$ itself a distribution, called the <span class="term">[prior](https://en.wikipedia.org/wiki/Prior_probability)</span>. It represents what we know or assume about $\theta$ before seeing the data. Once we observe the data $X_1,\dots,X_N$, we use <span class="term">[Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)</span> to update the prior, which gives the <span class="term">[posterior](https://en.wikipedia.org/wiki/Posterior_probability)</span> distribution $\theta\mid X_1,\dots,X_N$:

$$ P(\theta\mid X_1,\dots,X_N) = \frac{P(X_1,\dots,X_N\mid\theta)\,P(\theta)}{P(X_1,\dots,X_N)}, $$

that is, posterior $=$ likelihood $\times$ prior, divided by the probability of the data. The posterior plays a role similar to the sample distribution, and we often use its mean as an estimate of $\theta$ &mdash; though there are philosophical and mathematical differences that make the comparison imperfect. We'll focus less on philosophy and more on what's practically different, starting with an example.

### Bayesian inference for Bernoulli trials

Consider $X\sim\text{Bernoulli}(q)$, the outcome of a coin flip; for $N$ flips, $Y=\sum_{i=1}^N X_i$ is Binomial. In a Bayesian model we specify a prior for $q$, reflecting what we know about the coin &mdash; in reality, that it's very likely close to fair. To keep the calculation simple (and comparable to the frequentist approach), we'll instead use a prior where every value of $q$ is equally likely:

$$ q \sim \text{Uniform}(0,1), \qquad X\mid q \sim \text{Bernoulli}(q). $$

Our goal is the posterior $q\mid X_1,\dots,X_N$. Since the flips are independent, this depends only on $Y$, so we can equivalently find $q\mid Y$. The Uniform density is $f(q)=1$ (I'll use $f$ for a density and $P$ for a probability, though we'll sometimes have to mix them). By Bayes' theorem (or just the definition of conditional probability, twice),

$$ f(q\mid Y{=}y) = \frac{f(q,\lbrace Y=y\rbrace)}{P(Y=y)} = \frac{P(Y{=}y\mid q)f(q)}{P(Y=y)}, $$

where $f(q,\lbrace Y=y\rbrace)$ is the joint density/probability of $q$ and $Y$. Recall the Binomial pmf, $P(Y{=}y\mid q) = \binom{N}{y}q^y(1-q)^{N-y}$. All the $q$-dependence comes from this term, and since we're only trying to find a distribution *in* $q$, we can write

$$ f(q\mid Y{=}y) = C(N,y)\,q^y(1-q)^{N-y}, $$

where $C(N,y)$ is whatever constant makes $\int f(q\mid Y{=}y)\,dq = 1$, i.e. $C(N,y)^{-1} = \int_0^1 q^y(1-q)^{N-y}\,dq$. This is an example of a <span class="term">[Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)</span>: we say $q\sim\text{Beta}(a,b)$ if $q$ has density $f(q) \propto q^{a-1}(1-q)^{b-1}$, so here $q\mid Y \sim \text{Beta}(y{+}1,\,N{-}y{+}1)$. You don't need to memorize this, but it's worth looking up (Wikipedia, or the class Colab notebook) to see what the density looks like.

<div class="example" markdown="1">
#### Example (Laplace's rule of succession, the Bayesian view)

Recall the setting of <span class="term">[Laplace's rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession)</span>: we have $N$ independent Bernoulli trials, each with probability of success $q$, we observe $Y$ successes, and we want to estimate $q$. Laplace's rule of succession gives the estimator $\hat q_L = (Y{+}1)/(N{+}2)$, in place of $Y/N$.

<u>Question:</u> show that when $Y=N$ (all successes), $E[q\mid Y] = (N{+}1)/(N{+}2) = \hat q_L$.

<u>Solution:</u> here $C(N,N)^{-1} = \int_0^1 q^N\,dq = 1/(N{+}1)$, so

$$ E[q\mid Y] = (N{+}1)\int_0^1 q\times q^N\,dq = (N{+}1)\int_0^1 q^{N+1}\,dq = \frac{N+1}{N+2}. $$

So Laplace's rule of succession is the posterior mean under a uniform prior. This holds for every $Y$, not only $Y=N$: the posterior is $\text{Beta}(Y{+}1,\,N{-}Y{+}1)$, whose mean is $(Y{+}1)/(N{+}2)$.
</div>

To summarize, writing $Y$ for the data and $\theta$ for the parameter(s) ($q$ above), Bayes' theorem says

$$ P(\theta\mid Y) = \frac{P(Y\mid\theta)P(\theta)}{P(Y)}. $$

The pieces have names:

<ul>
<li>$P(\theta\mid Y)$ is the <span class="term">posterior</span> (a Beta distribution in the example above).</li>
<li>$P(Y\mid\theta)$ is the <span class="term">likelihood</span> (Binomial, above).</li>
<li>$P(\theta)$ is the <span class="term">prior</span> (Uniform, above).</li>
<li markdown="span">$P(Y)$ is the <span class="term">[evidence](https://en.wikipedia.org/wiki/Marginal_likelihood)</span>: the chance of observing the data, unconditional on the parameters (obtained by marginalizing over the prior).</li>
</ul>

While classical statistics aims to estimate $\theta$ and measure its uncertainty (the sample distribution), Bayesian statistics aims to compute the *entire posterior distribution*. From it we can get <span class="term">[point estimates](https://en.wikipedia.org/wiki/Point_estimation)</span> &mdash; e.g. the posterior mean, or the posterior maximum (a "maximum a posteriori" estimate) &mdash; but these are less central to the Bayesian approach than the posterior itself.

<details class="optional-section" open markdown="1">
<summary><strong>Aside: selecting priors with a Beta distribution</strong> <span class="optional-badge">(optional)</span></summary>

If $q\sim\text{Beta}(a,b)$, calculus gives

$$ E[q] = \frac{a}{a+b}, \qquad \text{var}(q) = \frac{ab}{(a+b)^2(a+b+1)}, $$

and (you can check by plotting the density in Python) as $a,b$ grow large, the Beta distribution approaches a Normal distribution.

<div class="example" markdown="1">
#### Example (selecting priors)

Suppose we're given a coin and want to determine whether it's biased from $N$ flips. We're pretty sure &mdash; like most coins &mdash; that it isn't: say, $95\%$ confident the bias is within $10\%$ of fair.

<u>Question:</u>
<ol type="a">
<li>Select a prior for $q$.</li>
<li>What's the posterior expectation if we flip the coin $5$ times and see $5$ heads?</li>
</ol>

<u>Solution:</u>
<ol type="a">
<li>A Beta distribution is approximately Normal for large enough $a,b$. Assuming this Normal approximation, we want
$$ P(\lvert q-0.5\rvert<0.1) = P(0.4<q<0.6) \approx 0.95. $$
Since we want the prior mean to be $1/2$, take $q\sim\text{Beta}(a,a)$. If $q$ is approximately Normal, this holds when
$$ 1.96\sqrt{\text{var}(q)} = 0.1 \implies \sqrt{\frac{a^2}{4a^2(2a+1)}} = \frac12\sqrt{\frac{1}{2a+1}} = \frac{0.1}{1.96} \implies a \approx 47.5. $$
</li>
<li>By Bayes' theorem,
$$ f(q\mid X_1,\dots,X_N) \propto q^Y(1-q)^{N-Y}\times q^{a-1}(1-q)^{a-1} = q^{Y+a-1}(1-q)^{N-Y+a-1}, $$
where $Y=\sum_i X_i$ &mdash; again, the $q$-dependence uniquely determines the distribution (up to normalization), so
$$ q\mid X_1,\dots,X_N \sim \text{Beta}(Y{+}a,\ N{-}Y{+}a). $$
With $N=5,Y=5,a\approx47.5$:
$$ E[q\mid X_1,\dots,X_N] = \frac{Y+a}{N+2a} = \frac{5+47.5}{5+2\times47.5} \approx 0.52. $$
</li>
</ol>
</div>

</details>

### Bayesian inference for a Normal mean

Suppose $X\mid\mu\sim\text{Normal}(\mu,\sigma^2)$, treating $\sigma^2$ as known and fixed, and consider Bayesian inference for $\mu$. Then $\overline X\mid\mu\sim\text{Normal}(\mu,\sigma^2/N)$. It's natural to take a Normal prior on $\mu$,

$$ \mu \sim \text{Normal}(m,s^2). $$

Then $\overline X$ is $\mu$ plus independent Normal noise, so $\mu$ and $\overline X$ are jointly Normal (check this!). Joint normality is what makes the calculation easy: it implies that $\mu\mid\overline X$ is Normal, and that its mean $E[\mu\mid\overline X]$ is a linear function of $\overline X$. Notice this is just a linear regression model for $\overline X$ with $\mu$ as the predictor (and slope $\beta=1$), so computing the posterior $\mu\mid\overline X$ amounts to "inverting" a linear regression model, something we've done before. Since $\mu\mid\overline X$ is Normal with a mean $b_1\overline X+b_0$ that is linear in $\overline X$, we only need its mean and variance:

$$ m = E[\mu] = b_1E[\overline X]+b_0, \qquad s^2 = \text{var}(\mu) = b_1^2\,\text{var}(\overline X) + \text{var}(\mu\mid\overline X), $$

where $b_1,b_0$ are the slope/intercept of the regression treating $\overline X$ as the predictor (so $b_0=E[\mu\mid\overline X{=}0]$). We know $E[\overline X]=m$ and $\text{var}(\overline X)=s^2+\sigma^2/N$. The slope $b_1$ comes from the covariance formula $\text{cov}(X,Y)=\beta_1\sigma_X^2$ (here $X=\mu$, $\beta_1=1$), giving $\text{cov}(\mu,\overline X)=s^2$, so

$$ s^2 = \text{cov}(\mu,\overline X) = b_1\big(s^2+\sigma^2/N\big) \implies b_1 = \frac{s^2}{s^2+\sigma^2/N}, $$

and hence $b_0 = m\big(1-b_1\big) = m\,\dfrac{\sigma^2/N}{s^2+\sigma^2/N}$. A bit more algebra gives $\text{var}(\mu\mid\overline X) = \dfrac{s^2\sigma^2}{s^2N+\sigma^2}$. In summary:

$$ \mu\mid\overline X \sim \text{Normal}\Big(b_1\overline X + m(1-b_1),\ \ \frac{s^2\sigma^2}{s^2N+\sigma^2}\Big). $$

Take note of what happens as $\sigma/s$ becomes very large or very small.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Naming the pieces

You want to estimate the fraction $q$ of a lake's fish that are trout. You assume each fish caught is an independent $\text{Bernoulli}(q)$ draw, you start from $q\sim\text{Beta}(3,3)$, and you catch $12$ fish of which $9$ are trout.

<ol type="a">
  <li>Which distribution is the prior, and which is the likelihood?</li>
  <li>Write down the posterior distribution of $q$.</li>
  <li>What is the posterior mean? How does it compare to the frequentist estimate $\hat q=9/12$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### How much does the prior matter?

Using the Normal-mean posterior $\mu\mid\overline X \sim \text{Normal}\big(b_1\overline X+m(1-b_1),\ \cdot\big)$ with $b_1 = s^2/(s^2+\sigma^2/N)$:

<ol type="a">
  <li>Compute $b_1$ when $\sigma^2=4$, $s^2=1$, and $N=8$, and write the posterior mean as a weighted average of $\overline X$ and $m$.</li>
  <li>What does $b_1$ approach as $N\to\infty$, and what does that say about the prior's influence?</li>
  <li>What happens as $s\to0$? Describe the resulting posterior in words.</li>
</ol>
</div>

</details>

## 6.2 Bayesian linear regression {#sec-6-2}

### The single-predictor case

Consider a single-predictor linear regression model

$$ Y_i = \beta X_i + \epsilon_i, \qquad i=1,\dots,N, $$

where $\beta$ is an unknown regression coefficient and the $\epsilon_i\sim\text{Normal}(0,\sigma_\epsilon^2)$ are independent, with $\sigma_\epsilon^2$ known. For convenience, assume $\mu_X=\mu_Y=0$ (zero marginal means for predictor and response). We give $\beta$ a prior distribution,

$$ \beta \sim \text{Normal}(0,\tau^2), $$

where $\tau^2$ represents our prior uncertainty about $\beta$. (The zero-mean assumption on the prior can be relaxed without too much extra work, but keeps the calculation simpler here.) We want the posterior distribution of $\beta$ given the observed data $(X_1,Y_1),\dots,(X_N,Y_N)$. Because the prior and the likelihood are Normal, the posterior is also Normal, so its mean and variance describe it completely. The posterior is $\beta\mid(X_1,Y_1),\dots,(X_N,Y_N)$ &mdash; but under our assumptions, this is the same as conditioning on the OLS estimator $\hat\beta$ and the $X$ points, since $\beta=\text{cov}(X,Y)/\text{var}(X)=E[XY]/E[X^2]$ (as $X,Y$ are mean zero), so

$$ \hat\beta = \frac{\sum_{i=1}^N X_iY_i}{\sum_{i=1}^N X_i^2}. $$

Since the $Y_i$ are Normal, $\hat\beta\mid(\beta,X_1,\dots,X_N)$ is Normal with

$$ E[\hat\beta\mid\beta,X_1,\dots,X_N] = \frac{\sum_i X_i\,E[Y_i\mid\beta,X_i]}{\sum_i X_i^2} = \frac{\sum_i X_i^2\,\beta}{\sum_i X_i^2} = \beta, $$

$$ \text{var}(\hat\beta\mid\beta,X_1,\dots,X_N) = \frac{\sum_i X_i^2\,\text{var}(Y_i\mid\beta,X_i)}{\big(\sum_i X_i^2\big)^2} = \frac{\text{var}(Y_i\mid\beta,X_i)}{\sum_i X_i^2} = \frac{\sigma_\epsilon^2}{N\hat\sigma_X^2}, $$

so $\hat\beta\mid\beta,X_1,\dots,X_N \sim \text{Normal}\big(\beta,\ \sigma_\epsilon^2/(N\hat\sigma_X^2)\big)$ &mdash; you might recognize this as exactly the sample distribution of the OLS estimator! Now we can use it to find the posterior of $\beta$ *given* $\hat\beta$: the key insight is that this is itself a linear regression model, with $\beta$ as the predictor and $\hat\beta$ as the response. "Inverting" this model as before (fill in the details yourself &mdash; see the exercises) gives

$$ \beta\mid(\hat\beta,X_1,\dots,X_N) \sim \text{Normal}\left(\hat\beta\,\frac{\tau^2}{\tau^2+\sigma_\epsilon^2/(N\hat\sigma_X^2)},\ \ \frac{\big(\sigma_\epsilon^2/(N\hat\sigma_X^2)\big)\tau^2}{\tau^2+\sigma_\epsilon^2/(N\hat\sigma_X^2)}\right). $$

### The nonlinear (feature-map) case

Now consider the general linear regression model with features (or, equivalently, many predictors &mdash; the difference here is mostly notational). Let

$$ Y_k = f(X_k) + \epsilon_k, \qquad f(x) = \sum_{i=1}^K \beta_i\phi_i(x), \qquad k=1,\dots,N, $$

where the $\epsilon_k\sim\text{Normal}(0,\sigma^2)$ are independent, and give each coefficient a prior $\beta_i \sim \text{Normal}(0,\tau_i^2)$. Assume $E[\phi_i(X)]=0$, and assume for simplicity that **$\sigma$ is known** (we don't need to estimate it, which simplifies things a lot).

A prior is called <span class="term">[conjugate](https://en.wikipedia.org/wiki/Conjugate_prior)</span> to a likelihood if the posterior belongs to the same family of distributions as the prior, so that observing data only changes the parameters of the distribution. We have already seen two examples: a Beta prior for Bernoulli trials gives a Beta posterior, and a Normal prior for a Normal mean gives a Normal posterior. Conjugate priors are convenient because the whole posterior is then described by a few numbers, which we can work out with algebra.

Normal priors are also conjugate here: the posterior distribution of the $\beta_i$ can be worked out analytically, and the marginal posterior of each $\beta_i$ is again Normal. Their means are determined by the following.

**Theorem (posterior mean for regression coefficients).** Define the $K\times K$ matrix

$$ \Omega_{i,j} = N\,\overline{\phi_i(X)\phi_j(X)} = \sum_{k=1}^N \phi_i(X_k)\phi_j(X_k), $$

and let $\bar\beta_i = E[\beta_i\mid y,X]$ denote the posterior mean of $\beta_i$. Then $\bar\beta_1,\dots,\bar\beta_K$ satisfy the $K$ equations

$$ \Big(\frac{\sigma}{\tau_i}\Big)^2\bar\beta_i + \sum_{j=1}^K \Omega_{i,j}\bar\beta_j = \sum_{k=1}^N \phi_i(X_k)Y_k. $$

A few remarks:

<ul>
<li>Here $N$ is the number of data points and $K$ is the number of features. If every $\tau_i$ is finite and strictly positive and $\sigma>0$, the system has exactly one solution, for any $N$ and $K$: the matrix $\Omega+\sigma^2\Lambda_0^{-1}$ in the matrix form below is positive definite, so it can be inverted. Without the prior term ($\tau_i\to\infty$, which gives least squares), a unique solution needs $\Omega$ to be invertible, which requires $N\ge K$.</li>
<li>Since $E[\phi_i(X)]=0$, the entries of $\Omega_{i,j}$ approximate covariances between the features. If we imagine the predictors drawn from some distribution and average over the data (both $X$ and $Y$),
$$ \frac{E[\Omega_{i,j}]}{N} = \text{cov}(\phi_i(X),\phi_j(X)), \qquad \frac1N\sum_{k=1}^N E[\phi_i(X_k)Y_k] = \text{cov}(Y,\phi_i(X)), $$
giving the "math world" version of the equations above (averaging over the data):
$$ \frac1N\Big(\frac{\sigma}{\tau_i}\Big)^2 E[\bar\beta_i] + \sum_{j=1}^K \text{cov}(\phi_i(X),\phi_j(X))\,E[\bar\beta_j] = \text{cov}(Y,\phi_i(X)). $$
As $N\to\infty$ with $K=2$, this is exactly the covariance-based system for the two-predictor regression coefficients from Unit 4 &mdash; except the role of the (previously fixed) regression coefficients is now played by $E[\bar\beta_j]$, the average of $\beta_j$ over both the posterior and the data.</li>
<li>The first term vanishes as either $\tau_i\to\infty$ (very broad priors) or $\sigma\to0$ (no conditional variance in $Y$) &mdash; in both cases $\bar\beta_i$ is determined entirely by the data, with no influence from the prior. As $\tau_i\to0$ or $\sigma\to\infty$, the prior dominates entirely and $\bar\beta_i\to0$.</li>
<li markdown="span">The first term is an example of <span class="term">[regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))</span>.</li>
</ul>

This theorem has an elegant matrix form. Let $A$ be the $N\times K$ design matrix with columns $\phi_1(X),\dots,\phi_K(X)$ (i.e. $A_{k,i}=\phi_i(X_k)$), so $\Omega=A^\top A$. Let $\Lambda_0=\text{diag}(\tau_1^2,\dots,\tau_K^2)$ be the diagonal matrix of prior variances, so that $\sigma^2\Lambda_0^{-1}=\text{diag}\big((\sigma/\tau_1)^2,\dots,(\sigma/\tau_K)^2\big)$, and define $M = \Omega+\sigma^2\Lambda_0^{-1}$. Then the equations above become $M\bar\beta=A^\top y$ (with $\bar\beta=(\bar\beta_1,\dots,\bar\beta_K)^\top$ and $y=(Y_1,\dots,Y_N)^\top$), i.e.

$$ \bar\beta = \big(A^\top A + \sigma^2\Lambda_0^{-1}\big)^{-1} A^\top y. $$

<div class="example" markdown="1">
#### Example (computing the posterior mean in Python)

Consider $f(x) = \beta_1X_1+\beta_2X_2$.

<u>Question:</u> generate simulated data and compute the posterior mean of $\beta$ for different choices of $\tau$.

<u>Solution:</u> see the companion Colab notebook.
</div>

### Posterior prediction

We'll mostly focus on how to use the posterior to make predictions (model assessment and hypothesis testing raise a different set of questions we won't cover). Recall $\hat y(x,D)$ is our prediction of $E[Y\mid X{=}x]$ using the model fit to data $D$. In the Bayesian setting, the <span class="term">[posterior predictive distribution](https://en.wikipedia.org/wiki/Posterior_predictive_distribution)</span> is the distribution of a new response $Y$ at a predictor value $x$, conditional on the observed data $D$: the distribution of $Y\mid X{=}x,D$. It describes the values a new observation could take, given what we have seen. It combines two sources of uncertainty: our uncertainty about the coefficients, described by their posterior distribution, and the noise $\epsilon$ in the new observation itself. We get it by taking the model's distribution for a new response, $Y\mid X{=}x,\beta\sim\text{Normal}\big(\sum_i\beta_i\phi_i(x),\,\sigma^2\big)$, and averaging it over the posterior distribution of the $\beta_i$. Its mean is our prediction:

$$ \hat y(x,D) = E[Y\mid D,X{=}x] = \sum_{i=1}^K E[\beta_i\mid D]\,\phi_i(x) = \sum_{i=1}^K \bar\beta_i\,\phi_i(x). $$

Writing $a = (\phi_1(x),\dots,\phi_K(x))^\top$, this is $\hat y(x,D) = a^\top\bar\beta = a^\top M^{-1}A^\top y$.

<div class="example" markdown="1">
#### Example (posterior predictions for a Fourier regression model)

Take the Fourier model $f(x) = \beta_0+\sum_{i=1}^K \big[\beta_i\sin\big(\frac{2\pi ix}{L}\big) + \alpha_i\cos\big(\frac{2\pi ix}{L}\big)\big]$, fit to data generated by adding noise to

$$ f_{\text{true}}(x) = \sin(2\pi\cdot3x) + 5\sin(2\pi\cdot6x^2) + 20x^2, \qquad x\in[0,1] $$

(note that for finite $K$, no choice of coefficients makes $f=f_{\text{true}}$ exactly). Assume $\beta_i,\alpha_i$ share the prior $\text{Normal}(0,\tau_i^2)$.

<u>Question:</u> compute the posterior mean $\hat y(x,D)$ with $K=50$ for (a) constant $\tau_i=\tau_0$, and (b) decaying $\tau_i=\tau_0/i$. In each case, try different $\tau_0$ and note whether the model overfits (high variance, low bias) or underfits (low variance, high bias).

<u>Solution:</u> see the companion Colab notebook.
</div>

The demo below fits a Fourier model with $K=12$ harmonics ($25$ coefficients) to $40$ equally-spaced points using the posterior mean $\bar\beta = (A^\top A + \sigma^2\Lambda_0^{-1})^{-1}A^\top y$, leaving the intercept unpenalized. Slide $\tau_0$ right and the prior stops mattering, leaving ordinary least squares; because these features are orthogonal on equally-spaced points ([Section 5.4](../unit5/#sec-5-4)) that does not blow up, but the fit does start chasing the noise. Slide $\tau_0$ left and every coefficient is squeezed toward zero until only the mean survives.

{% include_relative demos/ridge.html %}

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Shrinkage in the single-predictor case

The posterior mean of $\beta$ is $\hat\beta\,\tau^2/\big(\tau^2+\sigma_\epsilon^2/(N\hat\sigma_X^2)\big)$. Take $N=100$, $\hat\sigma_X^2=1$, $\sigma_\epsilon^2=4$, and $\hat\beta=2$.

<ol type="a">
  <li>Compute the posterior mean when $\tau^2=1$.</li>
  <li>Compute it again when $\tau^2=0.01$.</li>
  <li>Which prior pulls the estimate toward zero more, and why does that match the meaning of $\tau$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Limits of the posterior-mean equations

Consider $\big(\sigma/\tau_i\big)^2\bar\beta_i + \sum_{j=1}^K \Omega_{i,j}\bar\beta_j = \sum_{k=1}^N \phi_i(X_k)Y_k$.

<ol type="a">
  <li>What equation is left when $\tau_i\to\infty$, and what familiar estimator does it produce?</li>
  <li>What happens to $\bar\beta_i$ as $\tau_i\to0$?</li>
  <li>Why can this system be solved even when $K>N$, unlike ordinary least squares?</li>
</ol>
</div>

</details>

## 6.3 Regularization view of priors {#sec-6-3}

Recall that ordinary least squares finds the coefficients that minimize the training error. Writing $\hat y(x,D^{\text{train}})=\sum_{j=1}^K\beta_j\phi_j(x)$, this is the value of $\beta$ that minimizes the <span class="term">[loss function](https://en.wikipedia.org/wiki/Loss_function)</span>

$$ L(\beta) = \hat\sigma_{\text{train}}^2 = \frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}}\big(\hat y(X_i^{\text{train}},D^{\text{train}})-Y_i^{\text{train}}\big)^2. $$

A loss function is just any function of the parameters that we minimize to fit a model. Often it comes from some measure of how far the data are from the model's predictions. In the case of linear regression and least squares, this measure is actually a consequence of the underlying probability model. With Normal noise, the coefficients that make the observed data most likely are the ones that minimize the sum of squared residuals.

Here the data from Section 6.2 play the role of the training set, so $N_{\text{train}}=N$, and $\Omega_{j,l}=\sum_{i}\phi_j(X_i^{\text{train}})\phi_l(X_i^{\text{train}})$. Setting $\partial L/\partial\beta_j = 0$ for each $j$:

$$
\begin{aligned}
\frac{\partial L}{\partial\beta_j} &= \frac{2}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}}\big(\hat y(X_i^{\text{train}},D^{\text{train}})-Y_i^{\text{train}}\big)\phi_j(X_i^{\text{train}})\\
&= \frac{2}{N_{\text{train}}}\Big(\sum_{l=1}^K \Omega_{j,l}\beta_l - \sum_{i=1}^{N_{\text{train}}}Y_i^{\text{train}}\,\phi_j(X_i^{\text{train}})\Big) = 0
\end{aligned}
$$

Solving these $K$ equations for $\beta_1,\dots,\beta_K$ recovers the least-squares estimate $\hat\beta$. These are the posterior-mean equations above, but *without* the prior term $(\sigma/\tau_j)^2\bar\beta_j$.

The regularization view of priors starts with the loss function. Instead of thinking of that extra term as coming from a prior, we add a term to the loss function that penalizes overly flexible models. A common choice is

$$ L_\lambda(\beta) = \hat\sigma_{\text{train}}^2 + \lambda\sum_{j=1}^K \beta_j^2. $$

We write $L_\lambda$ to distinguish it from the least-squares loss $L$. It is the training error plus the penalty $\lambda\sum_j\beta_j^2$, and we write $\hat\beta^{\lambda}=(\hat\beta^{\lambda}_1,\dots,\hat\beta^{\lambda}_K)$ for its minimizer, the regularized (ridge) coefficients. Since $L_0=L$, $\hat\beta^{0}=\hat\beta$, but for $\lambda>0$ the minimizer is not $\hat\beta$. Minimizing $L_\lambda$ is called <span class="term">[ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)</span>, with $\lambda$ controlling how large a penalty we place on large $\beta_j$. Setting $\partial L_\lambda/\partial\beta_j=0$ gives

$$ N_{\text{train}}\lambda\,\beta_j + \sum_{l=1}^K \Omega_{j,l}\beta_l = \sum_{i=1}^{N_{\text{train}}}Y_i^{\text{train}}\,\phi_j(X_i^{\text{train}}), $$

which are exactly the posterior-mean equations with $(\sigma/\tau_j)^2=N_{\text{train}}\lambda$. More generally, with a separate penalty for each coefficient, minimizing

$$ L(\beta) + \frac{1}{N_{\text{train}}}\sum_{j=1}^K \Big(\frac{\sigma}{\tau_j}\Big)^2\beta_j^2 $$

recovers the posterior-mean equations exactly, for arbitrary per-coefficient priors $\tau_j$.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Ridge regression and Laplace's rule of succession

In words, describe how Laplace's rule of succession is doing the same thing as regularization.
</div>

<div class="exercise" markdown="1">
#### From code to a loss function

```python
x = np.random.normal(0, 1, 1000)
y = 3*x + np.random.normal(0, 1, 1000)
lam = 2
b = np.sum(x*y) / (np.sum(x**2) + 1000*lam)
```

<ol type="a">
  <li>Approximately what is <code>b</code>?</li>
  <li>Write down the loss $L_\lambda(b)$ that <code>b</code> minimizes.</li>
  <li>Approximately what would <code>b</code> be with <code>lam = 0</code>?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Choosing $\lambda$ from a table

A ridge regression model is fit to a training set for several values of $\lambda$ and scored on a test set:

<table class="prob-table">
<tr><th>$\lambda$</th><th>$\hat\sigma^2_{\text{train}}$</th><th>$\hat\sigma^2_{\text{test}}$</th></tr>
<tr><td>$10^{-4}$</td><td>0.21</td><td>0.95</td></tr>
<tr><td>$10^{-3}$</td><td>0.25</td><td>0.62</td></tr>
<tr><td>$10^{-2}$</td><td>0.31</td><td>0.48</td></tr>
<tr><td>$10^{-1}$</td><td>0.52</td><td>0.55</td></tr>
<tr><td>$1$</td><td>0.98</td><td>1.05</td></tr>
<tr><td>$10$</td><td>1.80</td><td>1.86</td></tr>
</table>

<ol type="a">
  <li>Which $\lambda$ would you choose, and why?</li>
  <li>Why does $\hat\sigma^2_{\text{train}}$ only go up as $\lambda$ grows?</li>
  <li>Which rows show overfitting, and which show underfitting?</li>
  <li>What does the smallest $\hat\sigma^2_{\text{test}}$ tell you, approximately, about $\sigma_\epsilon^2$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Completing the square for the ridge loss of a mean

For the model $Y=\mu+\epsilon$, the ridge loss is

$$ L_\lambda(\mu)=\frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}}\big(\mu-Y_i^{\text{train}}\big)^2+\lambda\mu^2. $$

Derive

$$ L_\lambda(\mu)=(1+\lambda)\Big(\mu-\frac{\overline Y}{1+\lambda}\Big)^2+\hat\sigma_Y^2+\frac{\lambda\,\overline Y^{\,2}}{1+\lambda}, $$

where $\hat\sigma_Y^2=\frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}}\big(Y_i^{\text{train}}-\overline Y\big)^2$.
</div>

<div class="exercise" markdown="1">
#### Bias and variance when shrinking toward $b$

Consider the single-predictor model $Y=\beta_1X+\epsilon$ with $E[X]=0$ and no intercept, fit to $N$ points with $\hat\sigma_X^2=\frac1N\sum_{i=1}^NX_i^2$ (treat the $X_i$ as fixed). The least-squares estimate satisfies $\hat\beta_1\approx\text{Normal}\big(\beta_1,\ \sigma_\epsilon^2/(N\hat\sigma_X^2)\big)$. Instead of shrinking toward $0$, minimize

$$ L_\lambda(\beta_1)=\frac1N\sum_{i=1}^N(\beta_1X_i-Y_i)^2+\lambda(\beta_1-b)^2 $$

for a number $b$. The minimizer is

$$ \hat\beta^{\lambda}_1=\frac{\hat\beta_1+(\lambda/\hat\sigma_X^2)\,b}{1+\lambda/\hat\sigma_X^2}. $$

Suppose the true slope is $\beta_1=3$. Give your answers in terms of $\lambda$, $b$, $\hat\sigma_X^2$, $\sigma_\epsilon^2$ and $N$.

<ol type="a">
  <li>Compute the bias and the variance of $\hat\beta^{\lambda}_1$.</li>
  <li>Compute the MSE of $\hat\beta^{\lambda}_1$.</li>
  <li>The fitted model predicts $\hat y(x,D)=\hat\beta^{\lambda}_1x$, and the true function is $f(x)=\beta_1x$. Compute $\text{MSE}_{\hat y(2,D)}=E_D\big[(\hat y(2,D)-f(2))^2\big]$.</li>
  <li>For a fixed $\lambda>0$, which $b$ gives the smallest MSE? Does the variance depend on $b$?</li>
</ol>
</div>

</details>

<details class="optional-section" id="sec-6-4" open markdown="1">
<summary><h2>6.4 The kernel trick</h2> <span class="optional-badge">(optional)</span></summary>

Computing $\hat y(x,D)$ is an example of <span class="term">[smoothing](https://en.wikipedia.org/wiki/Smoothing)</span>, or interpolation: given noisy measurements of a function $f(x)$, we want to "smooth out" the noise, e.g. by a weighted average of neighboring $y$ values. The following example shows how regression modeling relates to smoothing.

<div class="example" markdown="1">
#### Example (orthogonal empirical covariance matrix)

Consider the special case where $\Omega$ is diagonal, as it is for the Fourier model with equally-spaced $X_i$. Rescale the features so that $\sum_k\phi_i(X_k)^2=1$ for each $i$; then $\Omega_{i,j}=1$ if $i=j$ and $0$ otherwise, i.e. $\Omega=I$.

<u>Question:</u>
<ol type="a">
<li>Write $\hat y(x,D)$ in this case. In particular, express it as $\hat y(x,D) = \sum_{j=1}^N Y_j\,\kappa(x,X_j)$ for some function $\kappa(x,x^{\prime})$.</li>
<li>What does $\kappa(x,x^{\prime})$ look like for a Fourier model?</li>
</ol>

<u>Solution:</u>
<ol type="a">
<li>Here $\bar\beta_i = \dfrac{1}{1+(\sigma/\tau_i)^2}\sum_{j=1}^N \phi_i(X_j)Y_j$, so
$$ \hat y(x,D) = \sum_{i=1}^K \frac{\phi_i(x)}{1+(\sigma/\tau_i)^2}\sum_{j=1}^N \phi_i(X_j)Y_j = \sum_{j=1}^N Y_j\Big(\sum_{i=1}^K \frac{\phi_i(x)\,\phi_i(X_j)}{1+(\sigma/\tau_i)^2}\Big), $$
so $\kappa(x,x^{\prime}) = \displaystyle\sum_{i=1}^K \frac{\phi_i(x)\phi_i(x^{\prime})}{1+(\sigma/\tau_i)^2}$.</li>
<li>For the Fourier model with $L=1$,
$$ \kappa(x,x') = \sum_{i=1}^K \frac{\sin(2\pi ix)\sin(2\pi ix') + \cos(2\pi ix)\cos(2\pi ix')}{1+(\sigma/\tau_i)^2}. $$
Using $\cos(x)\cos(y)+\sin(x)\sin(y)=\cos(x-y)$, and $\cos(x-y)=\cos(y-x)$, this simplifies to
$$ \kappa(x,x') = \eta(\lvert x-x'\rvert) = \sum_{i=1}^K \frac{\cos\big(2\pi i\lvert x-x'\rvert\big)}{1+(\sigma/\tau_i)^2}. $$
</li>
</ol>
</div>

We don't actually need $\Omega$ to be diagonal to write $\hat y(x,D)$ in this "kernel" form &mdash; the general matrix formula $\hat y(x,D)=a^\top M^{-1}A^\top y$ has this structure too, though without a diagonal $\Omega$ we generally can't determine $\kappa$ without inverting a large matrix.

</details>

## Problems {#problems}

<div class="exercise" markdown="1">
#### Problem 6.1 &mdash; Polling with a Beta prior

Suppose you're polling Dartmouth students about political party affiliation, with (for simplicity) only two options: Democrat or Republican.

<ol type="a">
<li>Select a prior for $q$ as in the "selecting priors" example above (you'll need the Beta mean/variance formulas, though you don't need to memorize them). This step is subjective &mdash; the goal is to see how your own prior knowledge interacts with the data. Plot the prior in Python.</li>
<li>Now suppose $N$ students are surveyed and all are Democrats. Compute and plot the posterior for several values of $N$ on the same axes as the prior. Also plot the posterior mean and standard deviation as functions of $N$.</li>
<li>How large must $N$ be for you to be $95\%$ confident that at least $95\%$ of Dartmouth students are Democrats? Estimate this by hand (using the Beta mean/variance formulas again) or via simulation (sampling the posterior).</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 6.2 &mdash; Normal model and MSE

Consider the Bayesian Normal model with known $\sigma$, and suppose the true value of $\mu$ is $\mu_0$. Bayesian inference gives the posterior mean

$$ \hat\mu_P = E[\mu\mid\overline X] = m(1-b_1) + \overline X b_1 = m(1-b_1) + \hat\mu b_1, $$

where $\hat\mu=\overline X$, which we can treat as an estimator of $\mu$. In class we computed the usual MSE, $\text{MSE}_{\hat\mu_P} = E[(\hat\mu_P-\mu_0)^2]$, with the expectation over the data distribution.

An alternative definition of MSE is the expected squared difference between the true value and a sample $\mu$ *from the posterior*, averaged over all $\overline X$:

$$ \text{MSE}_{\hat\mu_P}' = E\big[E[(\mu-\mu_0)^2 \mid \overline X]\big], $$

where the inner expectation is over $\mu\mid\overline X$ (with $\mu_0$ fixed), and the outer is over the data distribution. Compute this quantity, interpret the terms, and compare the bias and variance to what we found for $\text{MSE}_{\hat\mu_P}$.
</div>

<div class="exercise" markdown="1">
#### Problem 6.3 &mdash; Bayesian linear regression

<ol type="a">
<li>Finish the derivation of the posterior $\beta\mid(\hat\beta,X_1,\dots,X_N)$ from Section 6.2 (the single-predictor case).</li>
<li>Generalize that result to the case where the prior mean is not zero.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 6.4 &mdash; Bayesian updating

Consider Bayesian inference for a Normal model with known variance, as in Section 6.1. With $N$ data points we get a posterior $\mu\mid\overline X$ (equivalently $\mu\mid\hat\mu$, since $\hat\mu=\overline X$). Now suppose we collect $K$ new points $X_{N+1},\dots,X_{N+K}$ and want to update the posterior. We could either:

<ul>
<li>compute the sample average $\overline X_{N+K} = \frac{1}{N+K}\sum_{i=1}^{N+K}X_i$ of all $N+K$ points and find the posterior $\mu\mid\overline X_{N+K}$ directly, or</li>
<li>use the posterior from the first $N$ samples as a new prior, then find a new posterior from just the $K$ new samples.</li>
</ul>

Determine whether these give the same answer.
</div>

<div class="exercise" markdown="1">
#### Problem 6.5 &mdash; Normal model with unknown variance

Consider a Bayesian Normal model where we also place a prior on $\sigma^2$. We can't use a Normal prior directly on $\sigma$ since it must be positive; a common fix is to put a Normal prior on $\ln\sigma$:

$$ \ln\sigma \sim \text{Normal}(l,\gamma^2), \qquad \mu \sim \text{Normal}(m,s^2), \qquad X_i\mid\mu,\sigma \sim \text{Normal}(\mu,\sigma^2), $$

with $\ln\sigma$ and $\mu$ independent under the prior. Pick values for $l$ and $\gamma^2$ and write code to sample from the marginal distribution of $\overline X$. Is it Normal? Compare it to a Normal density with the same mean and variance.
</div>

<div class="exercise" markdown="1">
#### Problem 6.6 &mdash; Ridge regression with one predictor

Consider the model $Y_i=\beta X_i+\epsilon_i$ with a mean-zero predictor and no intercept, and the ridge loss

$$ L_\lambda(\beta) = \frac1N\sum_{i=1}^N\big(Y_i-\beta X_i\big)^2 + \lambda\beta^2. $$

<ol type="a">
<li>Show that the minimizer of $L_\lambda$ is $\hat\beta^{\lambda}=\hat\beta\,\dfrac{\hat\sigma_X^2}{\hat\sigma_X^2+\lambda}$, where $\hat\beta=\sum_iX_iY_i\big/\sum_iX_i^2$ is the least-squares estimate and $\hat\sigma_X^2=\frac1N\sum_iX_i^2$.</li>
<li>Compute it when $\hat\beta=2$, $\hat\sigma_X^2=1$ and $\lambda=0.25$.</li>
<li>Compare it to the posterior mean of $\beta$ in the single-predictor case of Section 6.2. For which $\lambda$ are they equal? Write your answer in terms of $\sigma_\epsilon$, $\tau$ and $N$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 6.7 &mdash; Bias and standard error of a ridge estimate

Let $Y_1,\dots,Y_N$ be iid $\text{Normal}(\mu,\sigma^2)$, and consider the ridge loss for $\mu$,

$$ L_\lambda(\mu) = \frac1N\sum_{i=1}^N(\mu-Y_i)^2+\lambda\mu^2. $$

<ol type="a">
<li>Show that the minimizer of $L_\lambda$ is $\overline Y/(1+\lambda)$.</li>
<li>Compute the bias and $\text{se}^2$ of this estimator in terms of $\mu$, $\sigma$, $N$ and $\lambda$. For which value of $\mu$ is it unbiased?</li>
<li>Is it a consistent estimator of $\mu$ when $\lambda$ is fixed? How could $\lambda$ depend on $N$ so that it is consistent?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 6.8 &mdash; When does regularization help?

Continue Problem 6.6 with $\sigma_\epsilon^2=4$, $N=100$, $\hat\sigma_X^2=1$ and $\lambda=0.25$. Recall that the least-squares estimate has the sample distribution $\hat\beta\sim\text{Normal}\big(\beta,\ \sigma_\epsilon^2/(N\hat\sigma_X^2)\big)$.

<ol type="a">
<li>Find the bias and the variance of the ridge estimate $\hat\beta^{\lambda}$ from Problem 6.6 in terms of $\beta$.</li>
<li>Compute the MSE of $\hat\beta^{\lambda}$ and of $\hat\beta$ when $\beta=1$, and again when $\beta=0.1$.</li>
<li>In which case does regularization give the smaller MSE? Explain why in terms of bias and variance.</li>
</ol>
</div>
