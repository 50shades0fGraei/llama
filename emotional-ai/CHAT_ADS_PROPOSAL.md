# Chat Ads: privacy-first project proposal

## Executive summary

Chat Ads is an opt-in sponsored-discovery feature for conversational products.
When a user asks about a supported, general product topic, the host may show one
clearly labeled sponsored card beside—not inside—the assistant's answer. Matching
uses only the current message and a locally held, approved campaign catalog.

The value proposition is relevant discovery without psychological or demographic
profiling. It is a product hypothesis to validate, not a claim of market
dominance or guaranteed click-through performance.

## Product and client workflow

1. A user separately opts in to contextual sponsored suggestions. The default is
   off, and the host must honor opt-out immediately.
2. Paid clients submit a campaign with approved ad copy, an allowlisted general
   category, schedule, and optional HTTPS destination.
3. The host reviews the advertiser, claims, destination, and category before
   activation.
4. The local selector checks the current request for a category match. It returns
   at most one `SponsoredSuggestion`; otherwise it returns no ad.
5. The UI renders the card separately from the assistant's response, with the
   advertiser and sponsorship visible. The assistant does not present the ad as
   its own recommendation.

The repository includes the opt-in campaign data type and local selector in
[`CHAT_ADS.md`](CHAT_ADS.md). A paid-client dashboard, billing, approval queue,
and campaign scheduling are future host-application work, not implemented here.

## Data and trust boundaries

The proposal explicitly excludes:

- inferred or stored age, gender, location, occupation, personality, OCEAN traits,
  mood, emotional state, vulnerability, or psychological profile;
- ad targeting from conversation history, user identity, clicks, or cross-session
  behavior;
- sensitive-topic categories such as health, finances, politics, religion, or
  other intimate personal matters;
- hidden placement, clickbait, false urgency, fabricated endorsement, or
  optimization intended to pressure a user.

The current selector performs local matching on the current message only. It
makes no external request and records no click. Campaign data must be reviewed,
and the host UI must safely escape ad text. Any future analytics require a
separate privacy review and user-facing disclosure; the proposal favors aggregate
campaign-level reporting without user-level profiles.

## Business model and funding case

A pilot could test fixed campaign fees or time-limited sponsored placements in
approved categories. Pricing would not depend on inferred personal attributes or
emotional state. The funding case is to validate whether users value clearly
disclosed, opt-in contextual discovery and whether advertisers will pay for that
placement under these limits.

Market size, advertiser demand, conversion, and competitive advantage remain
unverified. Do not promise category leadership or superior click-through without
independent, reproducible pilot data.

## MVP and evaluation

1. Complete host-app consent controls, campaign review, category allowlist, and
   accessible sponsored-card rendering.
2. Run a small pilot with the feature off by default and a limited set of
   non-sensitive product categories.
3. Evaluate ad relevance and user trust alongside business performance.
4. Expand only if users understand the sponsorship and trust/safety measures
   remain within predeclared limits.

Primary measures: user-rated relevance, helpfulness, sponsorship comprehension,
opt-out rate, complaint rate, and the rate of irrelevant or sensitive-context
matches. If click-through is measured in a separately approved pilot, report it
only in aggregate, treat it as a secondary metric, and do not use it to build
individual profiles or intensify targeting. Compare against a no-ad baseline and
publish the test design and limitations.

## Pitch summary

> Chat Ads offers users optional, clearly labeled sponsored suggestions matched
> to the topic they are discussing right now. It gives advertisers a contextual
> placement while keeping personal profiles, emotional inference, and hidden
> persuasion out of the product. Funding supports a privacy review, host-app
> dashboard, and a controlled pilot to test user trust and commercial viability.
