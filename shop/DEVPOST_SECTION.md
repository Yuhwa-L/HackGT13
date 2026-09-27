## Applying it to agentic commerce: Snap to Shop

Shopping assistants are starting to act for us: spot a product in a photo, find it, put it in the cart, check out. Most
of them decide how far to go based on their own confidence, and that confidence is unreliable on real photos. A blurry,
dim or compressed picture can still produce a very sure-sounding answer that is simply wrong.

Snap to Shop puts our trust layer in charge of that decision instead. You snap a product; the assistant identifies it,
and an independent trust score decides what happens next:

- **Verified:** the assistant recommends picks tailored to your budget, style and past purchases, and checkout is one
  tap (up to a set limit).
- **Unsure:** it doesn't guess. It shows the two or three things it might be, compares them for you, and asks which
  one you meant. Checkout unlocks only after you choose, and still asks you to confirm.
- **Can't tell:** it asks for a better photo, with a concrete tip, and checkout stays locked.

The AI writes the conversation and the recommendations, but the rules are enforced in code: even if the assistant were
talked into "just buy it", the server refuses. The result is less friction when the system is genuinely right and
better questions when it isn't. In our tests, purchases in the one-tap tier were identified correctly
[TODO: result: TRUST-tier accuracy from data/shop/evaluation.json]. The products are real models at approximate prices, the shoppers are
fictional, and checkout is a mock with no payment data.
