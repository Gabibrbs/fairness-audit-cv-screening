# Fairness audit of a CV-screening model with Fairlearn

An audit of a fictional AI system that decides which job candidates go to interview. Built for Module 2 (Responsible AI) of the AI Specialisation at Tokio School.

> The notebook is written in European Portuguese, as it was submitted for the course.

## The setup

- **Data:** 5,000 synthetic candidates generated in the notebook (gender, years of experience, technical test score, education level, years of gap in the CV). No real personal data.
- **Built-in bias:** about 44% of candidates are truly qualified in both genders, but the historical recruiter decisions invited 17.5% of women and 39.7% of men. The bias was created on purpose: the history penalises CV gaps, which are more common among women, and also penalises women directly.
- **Model:** a logistic regression (scikit-learn) trained on 70% of the candidates to reproduce the historical decisions. **Gender is not one of its inputs.** The other 30% are used for testing.
- **Audit:** Fairlearn's `MetricFrame` compares selection rate, true positive rate and false positive rate by gender. `ThresholdOptimizer` (equal opportunity) is then applied as a post-training mitigation.

Because the data is synthetic, the true qualification of every candidate is known. That is what makes the bias measurable, and it is rarely possible in a real case.

## Results (from the run submitted for the course)

| Gap between genders | Before | After mitigation |
|---|---|---|
| Selection rate (demographic parity) | 16.0 pp | 6.5 pp |
| True positive rate (equal opportunity) | 31.8 pp | 7.0 pp |
| False positive rate, women / men | 0.9% / 2.8% | 24.1% / 5.3% |
| Accuracy against true qualification | 81.5% | 82.9% |

**Main finding:** the model discriminates against women even without seeing gender. The CV-gap feature acts as a **proxy** for gender (it has the strongest weight in the model, −3.957). Removing the gender column is not enough to make a model fair.

**The cost of the fix:** equal opportunity only equalises true positives. To raise women's true positive rate, their decision threshold went down, so more unqualified women also passed (false positives from 0.9% to 24.1%). There is no single fairness metric, and choosing one has trade-offs.

The notebook in this repository has no saved outputs. Run it to reproduce the figures.

## Limitations

- The data and the bias are synthetic.
- `ThresholdOptimizer` needs each candidate's gender at decision time, which can raise legal and privacy issues in a real system.
- Only gender was audited, not other groups or combinations.
- Security, robustness and energy use were not evaluated.

The course report also mapped the system to the EU AI Act (CV screening is high risk) and the GDPR (Article 22 on automated decisions).

## How to run

```
pip install -r requirements.txt
```

Open `fairness_audit_cv_screening.ipynb` and run all cells. The data is generated inside the notebook with a fixed seed.
