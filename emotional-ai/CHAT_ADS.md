# Chat Ads: opt-in contextual suggestions

Chat Ads is a separate workflow from assistant generation. It returns at most one
structured `SponsoredSuggestion` for the host application to render as a clearly
labeled ad card. It does not insert ad copy into the assistant's dialog or
disguise paid content as an independent recommendation.

The selector is disabled unless the host passes `consented=True` for the current
request. It performs local keyword matching against the current message and a
small allowlist of general product categories. It does not use conversation
history, inferred demographics, mood, psychological traits, clicks, or model
opinions, and it makes no external request. Health, finance, politics, and other
sensitive categories are not supported. The workflow does not track clicks.

```python
from emotional_ai import ChatAdCampaign, select_contextual_ad

suggestion = select_contextual_ad(
    current_user_message,
    campaigns=active_campaign_catalog,
    consented=user_opted_in_to_contextual_ads,
)
if suggestion is not None:
    # Render separately from assistant text, visibly labelled "Sponsored".
    render_sponsored_card(suggestion)
```

Campaigns require a supported category and bounded text; destination URLs, when
provided, must use HTTPS. The host must escape ad text in its UI, review copy for
accuracy, and honor opt-out immediately. Ads should not use clickbait, false
urgency, hidden persuasion, or claims unsupported by the advertiser.

This package provides the campaign data type and local selector, not a paid-client
dashboard, billing system, ad network, or campaign approval service. The host
application is responsible for those controls and must not add demographic or
mood targeting around this selector.
