# Lab Notes 8th Jan 2025 (OWASP)

## Unpacking OWASPs

**Background**

OWASP top 10 vulnerabilities for application security
SG Gov relies on OWASP for application security standards

- Ranked based on various factors like num_incidents
- regularly updated to reflect the current landscape
- LLM risks changed from 2023 to 2025
- However risks dont translate into concrete tests

**Today's Objective**
- standardize definitions
- familiarize with real word use cases and translate into monitoring modules
- Align Moonshot/monitoring work with industry best practices

Notes
- prompt injections continuously top risk
- new vulnerabilkites in 2025 like system prompt leakage, vector embedding weakness

**1. Prompt Injections**
   -  User prompts alters outputs in unintended ways
   -  PI and jailbreaking are interchangeable (jailbreaking is ignoring safety protocols)
   -  Impact: content policy safety violations etc

*Direct PI*: directly alters model behaviour (possibly unintentional)
*Indirect PI*: LLM accepts prompts from external sources, may contain hidden instructions embedded in external data (also possibly unintentional)

> SM
> *Level 1* - unsafe prompts
> *Level 2* - jailbreaking techniques (deliberate)

possible methods include roleplaying/obfuscation/payload splitting

> CH: need to categorize the different types of prompt injections
> from model perspective, is there a specific weakness/characteristic in the model I'm attacking?
> how can we **characterize** this problem
> this is so we can create unit tests, target individual vulnerabilities

> Eric on Dawn Song - we write attack prompts, write the defense prompts, then developers enforce using a very strong system prompt

One solution is to describe current techniques already in place and add to context (system prompt)



**2. Sensitive Information Disclosure**

Expose sensitive data, proprietary applications, confidential details
 - unauthorized data access
 - user interaction risks (users may unintentionally disclose sensitive data)

**3. Supply Chain**
risks affecting
- integrity of training data
- deployment platform

can result in biased outputs/security breaches/system failures

- e.g. vulnerable Python library in `PyPi`

**4. Data and Model Poisoning**

- data is manipulated in different stages of LLM lifecycle
- compromise model security, performance, ethical behaviour

**5. Improper Output Handling**

Insufficient validation/cleaning of outputs before being passed to downstream systems (possibly contain sensitive data)

Conditions exacerbating problem:
- excessive privileges to LLM
- indirect prompt injection
- weak input validation
- lack of monitoring
- insufficient usage controls (absence of rate limitign or anomaly detection)

**6. Excessive Agency**

LLMs have degree of agency (invoking tools)
- excessive functionality (tools have extra poorly designed functions)
- excessive permissions to LLMs
- excessive autonomy

**7. System Prompt Leakage**

- sensitive system prompts reveal critical details (API keys)
- disclose content rejection rules, allow attackers to bypass restrictions
- possibly disclose safety rules

**8. Vector Embedding Weaknesses**

vulnerbailities in embedding geenration/storage/retrieval allowing harmful contetn injection, output manipulation

> CH: what's the difference between risk 2 and risk 8
> risk 8 is specific to RAG in the embedding database
> risk is in the embedding database

> CH: what are some access control measures?
> SM: RBAC, different access control
> SM: researchers say very complicated no solutions

> CH: Is malicious content in LLM input?
> SM: when embedding comparision, retrieve the malicious content in the rag database
> CH: are there any persistent/consistent attacks?
> CH: Are there any attacks on the **systems** level?

> CH: Separate between Vulnerability of exploitation (risk) and Impact of Exploitation
> Currently risk is at the LLM end, not about output risk due to the LLM
> Should focus also on the causes and risks *caused by* the LLM outputs

> Director (on my left): Differentiate between business risks/tech risks?
> Tech risks come a few layers before business risks

> CH: how to structure the test dataset
> enumerating the weaknesses
> DTG risk/outcome based approach
> no need to demonstrate many different ways to output dangerous content
> **output risks** vs inherent model risks

> JF: Can classify risks based on which stage of LLM pipeline
> CH: Need to come back with archetypes (if not list is too long)
> will help us design the test dataset, if not will need to keep extending test set
> Need to better classify input and output risk

**9. Misinformation**

**10. Unbounded Consumption**

exploit the fact that LLM has high computational demands
  - high volume of queries
  - resource intensive queries


## Tests for OWASPs (Datasets and Modules)

- Problem statements for monitoring
- what does OWASP mean for moonshot

Based on industry feedback
- very specific concerns
- unsure what tests are relevant
- "Why are we running this?"
- RAG system owners
  - risks like hallucinations, reliability
  - dont care about toxicity
- most users dont know how to use redteaming module
  - still using old thinking of redteaming
  - how does it impact user
  - not intuitive for them, nobody really uses it
- Iterate Moonshot
  - Hallucinations
  - Existing Known vulnerability frameworks

## Final Comments

> CH: Break down into archetypes
> Better framing of the problem for testing
> Some way to structure coverage

> Director (on my left): Difficult to focus exclusive on hallucinations (due to policy focus etc)

> What makes a good redteaming agent good?
> SM:  Attack success rate, more efficient

> SM: Need to have well-defined objective, and include all possible known techniques

> CH: Need to get business risks to shape testing
> Maybe whitehouse trove of risks
> **Govtech 5 risks for apps**

> Director (on my left): If I'm a business owner, I look at this, so what?
> Relationship between OWASPS risks not clear

> Mapping need to do together
> Business can fit into prioritization
> Considering engineering, identify core risks first

> Must be improvement if we were to launch Moonshot as round 2
> Asking questions to the user - user testing first?
> "Are you interested in the OWASP test?"
> Users only know what they don't want

> CH: need to *validate first* before engineering

> Illustrative set of tests, get a sense of what would be useful for users

**Product side**
> questionaire on design partnerships
> Should be flexible; calibrate level of maturity design partners are at

**Labs side**
> provide some concrete examples
