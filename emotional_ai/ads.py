import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence
from urllib.parse import urlparse

AdCategory = Literal[
    "books",
    "cooking",
    "travel",
    "technology",
    "home",
    "garden",
    "sports",
    "music",
    "games",
    "office",
    "clothing",
    "pets",
]

_CATEGORY_TERMS = {
    "books": {"book", "books", "novel", "novels", "reading", "read"},
    "cooking": {"cook", "cooking", "recipe", "recipes", "kitchen"},
    "travel": {"travel", "trip", "flight", "hotel", "vacation"},
    "technology": {"laptop", "computer", "phone", "software", "technology"},
    "home": {"home", "furniture", "decor", "appliance"},
    "garden": {"garden", "gardening", "plant", "plants"},
    "sports": {"sport", "sports", "running", "team"},
    "music": {"music", "instrument", "album", "concert"},
    "games": {"game", "games", "gaming", "boardgame"},
    "office": {"office", "desk", "stationery", "workspace"},
    "clothing": {"clothing", "apparel", "shoes", "coat"},
    "pets": {"pet", "pets", "dog", "cat"},
}
_MAX_TEXT_LENGTH = 500
_MAX_FIELD_LENGTH = 120


@dataclass(frozen=True)
class ChatAdCampaign:
    campaign_id: str
    advertiser: str
    category: AdCategory
    text: str
    url: Optional[str] = None
    active: bool = True


@dataclass(frozen=True)
class SponsoredSuggestion:
    label: Literal["Sponsored"]
    campaign_id: str
    advertiser: str
    text: str
    url: Optional[str]


def select_contextual_ad(
    message: str,
    campaigns: Sequence[ChatAdCampaign],
    *,
    consented: bool,
) -> Optional[SponsoredSuggestion]:
    """Select one local, topic-matched ad only after explicit user consent."""
    if type(consented) is not bool:
        raise TypeError("consented must be bool")
    if not consented:
        return None
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if isinstance(campaigns, (str, bytes)):
        raise TypeError("campaigns must be a sequence of ChatAdCampaign values")

    terms = set(re.findall(r"\w+", message.casefold(), flags=re.UNICODE))
    for campaign in campaigns:
        _validate_campaign(campaign)
        if not campaign.active:
            continue
        if terms.intersection(_CATEGORY_TERMS[campaign.category]):
            return SponsoredSuggestion(
                label="Sponsored",
                campaign_id=campaign.campaign_id,
                advertiser=campaign.advertiser,
                text=campaign.text,
                url=campaign.url,
            )
    return None


def _validate_campaign(campaign: ChatAdCampaign) -> None:
    if not isinstance(campaign, ChatAdCampaign):
        raise TypeError("campaigns must contain ChatAdCampaign values")
    for name, value in (
        ("campaign_id", campaign.campaign_id),
        ("advertiser", campaign.advertiser),
        ("text", campaign.text),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if len(campaign.campaign_id) > _MAX_FIELD_LENGTH:
        raise ValueError("campaign_id must be 120 characters or fewer")
    if len(campaign.advertiser) > _MAX_FIELD_LENGTH:
        raise ValueError("advertiser must be 120 characters or fewer")
    if len(campaign.text) > _MAX_TEXT_LENGTH:
        raise ValueError("ad text must be 500 characters or fewer")
    if not isinstance(campaign.category, str) or campaign.category not in _CATEGORY_TERMS:
        raise ValueError("campaign category is not supported")
    if type(campaign.active) is not bool:
        raise TypeError("active must be bool")
    if campaign.url is not None:
        if not isinstance(campaign.url, str):
            raise TypeError("url must be a string or None")
        parsed = urlparse(campaign.url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("campaign URL must be a valid HTTPS URL")
