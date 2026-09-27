## Applying it to agentic commerce: the shopping assistant

Shopping assistants are starting to act for us, and most decide how far to go based on their own confidence. That
confidence is unreliable on real photos: a blurry or dim picture can still produce a very sure-sounding wrong answer.

Our shopping assistant puts the trust layer in charge of that decision instead. You snap a product; the assistant identifies it,
and an independent trust score decides what happens next:

- **Verified:** the assistant recommends picks tailored to your budget, style and past purchases, and checkout is one
  tap (up to a set limit).
- **Unsure:** it doesn't guess. It shows the two or three things it might be, compares them for you, and asks which
  one you meant. Checkout unlocks only after you choose, and still asks you to confirm.
- **Can't tell:** it asks for a better photo, with a concrete tip, and checkout stays locked.

The AI writes the conversation and the recommendations, but the rules are enforced in code: even if the assistant were
talked into "just buy it", the server refuses. Every purchase carries a tamper-evident record of why it was allowed. On real phone photos we took ourselves, all 7 one-tap purchases were the right
item, and none of the 4 items outside the catalog was ever trusted; an assistant going on its own confidence would have
bought a laptop as a keyboard. The products are real models at approximate prices, the shoppers are
fictional, and checkout is a mock with no payment data.
