# Clinical decision support for pharmacologic management of treatment-resistant depression with augmented large language models

## Background

We evaluated whether a large language model could assist in selecting psychopharmacological treatments for adults with treatment-resistant depression.

## Methods

We generated 20 clinical vignettes reflecting treatment-resistant depression among adults based on distributions drawn from electronic health records. Each vignette was evaluated by 2 expert psychopharmacologists to determine and rank the 5 best next-step pharmacologic interventions, as well as contraindicated or poor next-step treatments. Vignettes were then presented in random order, permuting gender and race, to a large language model (Qwen 2.5:7B), augmented with a synopsis of published treatment guidelines. Model output was compared to expert rankings, as well as to those of a convenience sample of community clinicians and an additional group of expert clinicians.

## Results

The augmented model prioritized the expert-designated optimal choice for 114/320 vignettes (35.6 %, 95 % CI 30.6 %-41.0 %; Cohen's kappa = 0.34, 95 % CI 0.28-0.39). There were no vignettes for which any of the model choices were among the poor or contraindicated treatments. Results were not meaningfully different when gender or race of the vignette was permuted to examine risk for bias. A sample of community clinicians identified the optimal treatment choice for 12/91 vignettes (13.2 %, 95 % CI: 7.7-21.6 %; Cohen's kappa = 0.10, 95 % CI 0.03-0.18), while an additional group of expert psychopharmacologists identified optimal treatment for 9/140 (6.4 %, 95 %CI: 3.4-11.8 %; Cohen's kappa = 0.03, 95 % CI 0.01-0.08).

## Conclusion

An augmented language model demonstrated moderate agreement with expert recommendations and avoided contraindicated treatments, suggesting potential as a tool for supporting complex psychopharmacologic decision-making in treatment-resistant depression.
