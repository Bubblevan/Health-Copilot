# M3 claim-support fixture audit

## m3s-007 wrong citation binding repair

- Old fixture claim: `规律测量血压有助于了解血压是否偏高。`
- Old cited source: `who-hypertension-03-silent`
- Old observed evidence: `who-hypertension-03-silent`,
  `cdc-high-blood-pressure-managing-01-monitoring`
- Old gold: `UNSUPPORTED`

The old annotation was invalid: the WHO card itself says that measuring blood
pressure is an important way to learn whether someone may have hypertension, so
it materially supported the claim.

- New fixture claim: `家庭自测血压时应至少测量两次，每次间隔 1 到 2 分钟。`
- New cited source: `who-hypertension-03-silent`
- New observed evidence: `who-hypertension-03-silent`,
  `cdc-high-blood-pressure-measuring-02-repeat`
- New gold: `UNSUPPORTED`

The new cited WHO card does not support a repeated-reading count or interval.
The observed CDC repeat-reading card does support the claim, but the claim does
not cite it. This is a reviewed annotation repair, not a change made in response
to a model prediction.
