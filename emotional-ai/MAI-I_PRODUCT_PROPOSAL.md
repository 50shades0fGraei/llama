# MAI-I: social connection and shared-browsing product proposal

## Vision

MAI-I is a proposed social product combining opt-in dating, shared browsing, and
small online hangout spaces—a virtual internet café where people choose what to
share and when to interact. The initial business hypothesis is that people value
an easy way to discover a shared topic and spend time together online.

This is a concept proposal, not an implemented Messenger or Vercel integration.
Platform access, audience demand, revenue, and market size still need validation.

## Product principles

- **People control their profiles.** Use interests and preferences people choose
  to provide. Do not scrape Facebook profiles, private messages, friend graphs, or
  browsing history. Any future official platform integration must use permitted
  APIs, minimum scopes, clear consent, and deletion controls.
- **No psychological or mood profiling.** Do not infer OCEAN traits, emotional
  vulnerability, personality, or dating intent from tone or private conversations.
  Do not target ads or matches with those inferences.
- **AI practice is clearly role-play.** A user may practice a conversation or
  plan a date with a fictional AI partner. The AI must not impersonate a real
  match or claim to be a digital copy of a person. Feedback can offer optional
  conversation ideas; it must not predict attraction, compatibility, or a date's
  “success rate.”
- **Shared spaces are opt-in.** A person joins a room and intentionally shares a
  screen, link, or topic. Presence is visible, sharing is reversible, and no
  background monitoring reveals what people are doing elsewhere.
- **Sponsored content is distinct.** Ads remain optional, clearly labeled,
  contextual to a general current topic, and separate from advice or dating
  recommendations. See [`CHAT_ADS.md`](CHAT_ADS.md).

## Proposed experience

### 1. User-controlled profile and matching

Users create a profile with fields they choose to disclose, select their own
matching preferences, and can edit or remove those fields. Matching should be
based on compatible, user-selected criteria and mutual interest, with a clear
explanation of the criteria used. Do not create hidden scores from profile
scraping, psychological traits, or inferred demographics.

### 2. Conversation and date practice

The user can choose a fictional practice scenario, such as starting a
conversation, planning a low-pressure activity, or discussing a shared interest.
The assistant can suggest questions, help compare date plans, and remind the user
to respect boundaries. The mode is visibly AI role-play, not an assessment of
another person or a forecast of romantic success. Session retention is off by
default; the user can explicitly save or delete notes.

### 3. Virtual café and shared wall

People can create or join rooms with a topic, invite link, participant list, and
clear sharing state. A discovery wall can show only room cards that participants
have chosen to make discoverable, such as “open room: cooking discussion.”
Joining a room never silently opens a participant's screen or private browsing.
Room members control screen/link sharing, can stop sharing at any time, and can
block or report others.

### 4. Messenger and web integration

Start with a standalone web prototype and invite links. Treat Facebook Messenger
as a possible distribution or sharing channel only after checking current
platform terms, available permissions, and review requirements. Do not assume the
product can read Messenger conversations or profiles. The existing repository
does not contain the user's Vercel project or credentials, so no external project
has been changed.

## Safety and privacy requirements

- Make dating age eligibility explicit and apply the host's legal and platform
  requirements before launch.
- Provide blocking, reporting, moderation, rate limits, and clear room ownership.
- Make consent specific to profile fields, room visibility, screen sharing, AI
  practice, ads, and any optional analytics; one consent must not imply another.
- Avoid collecting sensitive attributes unless strictly necessary, lawful, and
  explicitly chosen by the user. Do not infer or use them for ranking.
- Set retention periods, provide account export/deletion, and keep private rooms
  and profile data inaccessible to advertisers.
- Do not use engagement, emotional intensity, or inferred dating vulnerability
  as an optimization target.

## MVP and validation plan

1. **Discovery:** interview prospective users and test whether shared browsing or
   dating practice solves a real problem. Obtain informed research consent.
2. **Café prototype:** build private rooms, invite links, explicit presence, and
   user-controlled link/screen sharing. Keep dating and ads out of the first test.
3. **Practice mode:** add clearly fictional, user-directed conversation practice
   with no compatibility or success scoring.
4. **Dating pilot:** add user-entered profiles and explainable, mutual matching
   only after safety, moderation, and privacy reviews.
5. **Distribution and revenue:** assess Messenger sharing and opt-in contextual
   sponsored cards separately; do not make platform access or ad revenue a launch
   assumption.

Evaluate room usefulness, return use, user-reported comfort, consent
comprehension, successful blocking/reporting, and safety incidents. If testing
sponsored content, track only disclosed, privacy-reviewed aggregate metrics and
keep user trust and opt-out rates as guardrails. Do not claim a multi-billion
dollar opportunity or market dominance without independently verified research.

## Early pitch

> MAI-I helps people spend time together online: they can discover opt-in topic
> rooms, share a page or screen intentionally, and practice conversations with a
> clearly identified fictional AI. The product is designed around user-controlled
> data and mutual consent. A staged pilot will test whether the shared-café
> experience creates enough value to support later dating and creator or sponsor
> features.
