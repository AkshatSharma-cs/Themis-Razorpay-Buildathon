# Themis Mobile App Roadmap

> **Planning only:** Nothing in this document is implemented. This is a roadmap for future work. No permission-requesting code should be written as part of this task.

## Recommended Direction

The current product is a vanilla JavaScript frontend backed by FastAPI. The existing backend can be reused unchanged by a mobile client through the documented `/v1` HTTP API.

| Option | Strengths | Tradeoffs |
|---|---|---|
| Native Android with Kotlin + native iOS with Swift | Best platform integration, permission UX, accessibility, performance, and store compliance | Two codebases and two teams or larger maintenance cost |
| React Native or Flutter | One client codebase, good native-module support, faster than two native apps | Requires a new mobile UI layer and platform-specific modules for sensitive signals |
| Capacitor/Cordova shell around the web app | Fastest installable prototype; reuses the existing HTML/CSS/JavaScript UI | Limited access to restricted device data, weaker native UX, and permission plugins still need platform review |

**Recommendation:** start with Capacitor as a thin shell for the fastest installable demo, with no new permissions. The team is currently web-first, so this preserves the existing UI while calling the FastAPI service. If device signals prove valuable, add focused native modules behind the same web-to-API boundary rather than rewriting the backend. A longer-term consumer product should consider React Native or Flutter if a richer cross-platform mobile UI is needed; a high-compliance banking product may ultimately justify separate Kotlin and Swift clients.

## What Stays The Same

The mobile client would reuse the existing FastAPI backend and `/v1` endpoints from `API.md`. The scoring and decision logic, `DefenseAction` invariant, narration boundary, and SQLite/hash-chain audit design would remain unchanged from the client's perspective. A new native or hybrid client layer would submit the same transaction fields and display the same score, action, explanation, release flow, and audit results.

The first Capacitor phase would reuse the existing SMS checker as a manual text-entry workflow. It would not silently gain device access just because it runs inside a mobile shell.

## Platform Permission Reality

### Android

Android could expose additional signals through permissions such as:

- `READ_SMS` and/or `RECEIVE_SMS`: automatically inspect incoming messages for scam pressure using the existing SMS-check logic.
- `READ_CALL_LOG`: detect whether a call was active or recent around a UPI payment and feed `call_overlap_flag` and `call_minutes` features.

These permissions are heavily restricted on modern Android. Apps that are not the user's default SMS or phone handler generally cannot assume access merely by adding a manifest entry. Google Play requires a permissions declaration and review for restricted SMS and call-log permissions, and most apps are rejected unless SMS or calling is a core function. Any Android implementation must plan for the Play Console permissions declaration form, policy review, user-facing justification, and possible rejection. This is a product and distribution constraint, not just an engineering task.

### iOS

iOS does not allow third-party apps to read SMS content or call logs. This is an OS-level restriction with no general exception for security or banking apps.

The realistic iOS alternatives are:

- Let users manually paste or forward suspicious SMS text into the existing SMS checker.
- Investigate an `ILMessageFilterExtension`, which can classify incoming messages as junk without giving the app access to the user's full SMS history. This is a message-filtering path, not unrestricted SMS reading.
- Do not promise call-log-based `call_overlap_flag` or `call_minutes` on iOS; those signals are unavailable to ordinary third-party apps.

## Permission UX

Ask just in time, only when a feature needs the signal. Before an Android runtime permission prompt, show a short explanation such as: "Themis can check a suspicious message on this device so you do not need to copy it manually." Explain what is collected, where it is processed, and that the feature works without permission through manual entry. Request each permission separately rather than asking for SMS and call-log access at first launch.

On Android, use runtime permission prompts after the explanation and provide a useful fallback when the user declines. If a permission is permanently denied, link to the app's settings guidance rather than repeatedly prompting. On iOS, use the limited system-supported message-filter extension flow and manual paste/forwarding; there is no equivalent prompt that unlocks SMS history or call logs.

## Privacy and Compliance

SMS content, call metadata, payment details, and risk decisions are sensitive user data. A real app would need a clear privacy policy, explicit purpose limitation, data retention rules, access controls, deletion support, and auditability.

Prefer on-device processing for raw SMS and call signals where practical. Do not upload raw message content unless it is necessary, clearly disclosed, protected in transit, and covered by a defensible retention policy. Send only minimized derived signals to the backend when possible, such as a pressure-signal result rather than the entire message.

If Themis ever handles real Indian users' financial or personal data, the team should assess obligations under India's Digital Personal Data Protection Act, 2023 (DPDP Act), including notice, consent or another lawful basis, purpose limitation, security safeguards, retention/deletion, user rights, and data-principal requests. Obtain current legal and platform-policy advice before production release; this roadmap is not legal advice.

## Phased Plan

### Phase 1: Installable web shell

Package the existing web frontend with Capacitor. Add mobile navigation and offline-friendly loading states, but request no new permissions. Keep the current manual SMS checker and call the existing Render API using the same `/v1` contract.

### Phase 2: Android device signals

Prototype an Android-native module for SMS and call-context signals only after defining the privacy model. Process raw content on-device where possible, convert it to minimized features, and pass derived values to the existing API. Complete the Google Play restricted-permission declaration and review before promising automatic scanning in the store build.

### Phase 3: iOS-compatible message filtering

Add an iOS `ILMessageFilterExtension` investigation and keep manual paste/forwarding as the universal fallback. Do not attempt unrestricted SMS-history or call-log access. Validate Apple's extension rules, message classification limits, and user disclosure requirements.

### Phase 4: Production hardening

Replace demo API credentials and ephemeral local audit storage with production identity, durable storage, observability, secure secret management, consent records, retention controls, threat modeling, and platform-specific compliance review. These are future work items, not implemented by this document.
