# Probe Set: 30 messy-input test cases

Ground truth is known by construction: each case names a real catalog product and its true attributes.
Write the utterance a person wanting THAT product would actually type. Do not use catalog vocabulary.

For each case fill in: `utterance`, then the `expected_parse` you'd want a parser to produce.
Tier rule: HARD = brand/category/department/material/size/price (three-valued: match/contradict/SILENT).
SOFT = colour, use-case, season, vibe, comfort/fit/durability claims.

---

## 01. `B00D8G0WX8` — stratum A situation/occasion-driven

**ASICS Mens Gel-Resolution 5**

- category path: `Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Tennis & Racquet Sports`
- coarse bucket: `Athletic Tennis & Racquet Sports` (74 siblings)
- brand: `ASICS` | department: `Mens` | price: `— (absent)` | ratings: 268
- features: ["100% Synthetic", "Imported", "Rubber sole", "Lace-up tennis shoe featuring Flexion Fit upper with rearfoot and forefoot GEL cushioning systems", "Memory foam lined collar and mold to heel"]
- details: {"Package Dimensions": "13.1 x 7.6 x 4.8 inches; 15.4 Ounces", "Item model number": "Gel-Resolution 5-M", "Department": "Mens", "Date First Available": "January 10, 2014", "Manufacturer": "ASICS"}

> **What this tests:** Situation-driven. The user will describe an OCCASION or TRIP, not attributes. Tests inference over world knowledge.

```yaml
utterance: "i want a men's shoe for playing tennis, preferably something that has decent cushioning so it's comfortable. i also want the shoe to be quite grippy so that i don't slip on the court. hmm does the material of the rest of the shoe matter? i guess not? idk"            # <- write the human sentence here
expected_parse:
  hard:
    category: tennis-shoes        # NOT "shoes": 89 products vs 9,505, and "shoes" misses the target
    department: mens              # three-valued - keep products SILENT on department
  soft: [cushioning, grippy, comfortable]
  declined: [material]            # explicitly doesn't care != never mentioned
  discard: ["so i don't slip on the court", "hmm does the material of the rest of the shoe matter? i guess not? idk"]
  ceiling: achievable             # bucket 74; constraints are discriminative
```

---

## 02. `B00E4N07B6` — stratum A situation/occasion-driven

**RYKA Women's Tenacity Cross-Trainer Shoe**

- category path: `Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Fitness & Cross-Training`
- coarse bucket: `Athletic Fitness & Cross-Training` (136 siblings)
- brand: `Ryka` | department: `—` | price: `82.32` | ratings: 748
- features: ["100% Synthetic", "Imported", "Man made sole", "Shaft measures approximately 4\" from arch", "Heel measures approximately 0.75\""]
- details: {"Package Dimensions": "12 x 8 x 4 inches; 10 Ounces", "Item model number": "TENACITY-W", "Date First Available": "February 15, 2014", "Manufacturer": "RYKA (Caleres, Inc)"}

> **What this tests:** Situation-driven. The user will describe an OCCASION or TRIP, not attributes. Tests inference over world knowledge.

```yaml
utterance: "i'm looking for a shoe that is good for all purpose usage, like if i do a mixture of sports like lifting weights, running,  maybe maybe if i want to do racket sports as well. i'm a woman."            # <- write the human sentence here
expected_parse:
  hard:
    category: cross-training-shoes  # EMERGENT: no single word says this
    department: womens              # three-valued REQUIRED - target has NO Department field
  soft: [versatile, weightlifting, running, racquet-sports, supportive]
  declined: []
  discard: []
  ceiling: achievable
  notes: |
    Conjunction-implies-category. Literal terms ("lifting weights running racket
    sports") rank the target 7,232. The inferred category ranks it 9.
    The individual sports are DISTRACTORS as hard constraints, harmless as soft.
```

---

## 03. `B00KZIV0Q0` — stratum A situation/occasion-driven

**Merrell Women's Vapor Glove 2 Barefoot Trail Running Shoe**

- category path: `Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Running > Trail Running`
- coarse bucket: `Running Trail Running` (138 siblings)
- brand: `Merrell` | department: `womens` | price: `— (absent)` | ratings: 1,549
- features: ["100% Textile/Synthetic", "Imported", "Rubber sole", "Barefoot-style trail runner in mesh and TPU with traditional lace closure, breathable mesh lining, and integrated microfiber soft footbed", "0mm cushioning with 0mm heel-to-toe drop"]
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "11 x 7 x 4 inches; 4.7 Ounces", "Item model number": "J03918", "Department": "womens", "Date First Available": "January 12, 2015", "Manufacturer": "Merrell"}

> **What this tests:** Situation-driven. The user will describe an OCCASION or TRIP, not attributes. Tests inference over world knowledge.

```yaml
utterance: "looking for a pair of women's shoes for running in the forests and mountains. trail running is it called? i prefer stuff from a more reputable brand and something well-rated."            # <- write the human sentence here
expected_parse:
  hard:
    category: trail-running-shoes
    department: womens
  soft: [trail, outdoor, mountain]
  quality:                        # maps to numeric fields, not text
    brand_reputation: high        # -> no catalog field; proxy via rating_number
    well_rated: true              # -> average_rating / rating_number (100% coverage)
  declined: []
  discard: ["trail running is it called?"]   # self-correction, not a constraint
  ceiling: rank-6                 # 13 candidates satisfy; target 6th by popularity
  notes: |
    "reputable brand" and "well-rated" are the user explicitly ASKING FOR THE PRIOR.
    Neither maps to a text attribute; both map to rating_number/average_rating.
    Under-specified: the target's defining trait (barefoot, 0mm heel-to-toe drop)
    is never mentioned, so rank 1 is not reachable from this utterance.
```

---

## 04. `B07FDB7GMZ` — stratum A situation/occasion-driven

**adidas Men's Adilette Shower Slide**

- category path: `Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Sport Sandals & Slides`
- coarse bucket: `Athletic Sport Sandals & Slides` (132 siblings)
- brand: `adidas` | department: `Mens` | price: `28.82` | ratings: 109,066
- features: ["100% Synthetic", "Imported", "Rubber sole", "Shaft measures approximately low-top from arch", "adidas men's Slide Sandal"]
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "8 x 4 x 7 inches; 2.45 Pounds", "Item model number": "AQ1701", "Department": "Mens", "Date First Available": "July 1, 2004", "Manufacturer": "Adidas"}

> **What this tests:** Situation-driven. The user will describe an OCCASION or TRIP, not attributes. Tests inference over world knowledge.

```yaml
utterance: "i need a pair of waterproof pair of slides so i can wear them to the beach or shower."            # <- write the human sentence here
expected_parse:
  hard:
    category: slides              # sport sandals / slides
    department: null              # NOT EXTRACTABLE - user never states gender; target is Mens.
                                  # Filtering either way is wrong. Must stay unset.
  soft: [waterproof, beach, shower, quick-dry]
  declined: []
  discard: []
  ceiling: rank-1                 # target has 109,066 ratings - most popular in its bucket
  notes: |
    "waterproof" is an INFERENCE about the material. The product text never says it
    (only "100% Synthetic", "Rubber sole"). Pure vocabulary gap.
    Popularity prior alone solves this case - useful as a control.
```

---

## 05. `B07K3SMCXL` — stratum A situation/occasion-driven

**COSKAKA Womens One Piece Swimsuits Off Shoulder Flounce Ruffled with Removable Straps Padded Swimwear Bathing Suits**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Swimsuits & Cover Ups > One-Pieces`
- coarse bucket: `Swimsuits & Cover Ups One-Pieces` (234 siblings)
- brand: `COSKAKA` | department: `Womens` | price: `— (absent)` | ratings: 361
- features: ["Spandex,Nylon", "Imported", "Drawstring closure", "\u2600About SIZE:This one piece swimsuits have specially designed different US sizes to choose: S(US 4-6),M(US 8-10), L(US 12-14), XL(US 16), XXL(US 18).", "\u2600Fully Lined Design:BODY--85% Nylon & 15% Spandex\uff1b LINING--92% Nylon & 8% Spandex.Top soft material, conservative hip and leg coverage design. Swimsuit would not get see-through wh
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "10 x 8 x 0.5 inches; 8 Ounces", "Department": "Womens", "Date First Available": "October 25, 2018", "Manufacturer": "Coskaka"}

> **What this tests:** Situation-driven. The user will describe an OCCASION or TRIP, not attributes. Tests inference over world knowledge.

```yaml
utterance: "i need a stretchy bathing suit for swimming, something a lil sexy but also comfortable with things like padding and support"            # <- write the human sentence here
expected_parse:
  hard:
    category: swimwear            # user says "bathing suit" - does NOT specify one-piece
    department: womens            # weakly inferred, not stated
  soft: [stretchy, padded, supportive, comfortable, flattering]
  declined: []
  discard: []
  ceiling: NOT-IDENTIFIABLE       # 234 candidates, target ranks 40 by popularity
  notes: |
    "stretchy" -> spandex/nylon is a vocabulary bridge the parser must make.
    "a lil sexy" is pure vibe: no contradiction set, soft-only, near-zero IDF.
    The user never says "one-piece", so nothing separates this from 233 siblings.
    EXPECTED TO FAIL. Keep it - a probe set with no unsolvable cases is lying.
```

---

## 06. `B089QYV4SD` — stratum A situation/occasion-driven

**Women Light Rain Jacket Waterproof Active Outdoor Trench Raincoat with Hood Lightweight Plus Size for Girls**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Trench, Rain & Anoraks > Raincoats`
- coarse bucket: `Rain & Anoraks Raincoats` (49 siblings)
- brand: `SOMIDE` | department: `Womens` | price: `— (absent)` | ratings: 493
- features: ["waterproof material", "Imported", "Drawstring closure", "Hand Wash Only", "\u3010Waterproof raincoat design\u3011Waterproof material, long sleeves. The front zip and wind buckle design, drawstring hoodie and drawstring hem keep the raincoat tightly on your body, which can fully protect you, and the polyester spandex fabric has a higher density to keep you dry. With this waterproof raincoat, you 
- details: {"Product Dimensions": "13.78 x 11.02 x 1.18 inches; 1.31 Pounds", "Department": "Womens", "Date First Available": "August 18, 2020", "Manufacturer": "SOMIDE"}

> **What this tests:** Situation-driven. The user will describe an OCCASION or TRIP, not attributes. Tests inference over world knowledge.

```yaml
utterance: "i'm a big sized female, looking for a raincoat to keep me out of the water when it's pouring. hopefully something that has some good designs to prevent the water from coming in especially at the seams and zippers and stuff"            # <- write the human sentence here
expected_parse:
  hard:
    category: raincoat
    department: womens            # "female" - stated
    size: plus-size               # "big sized" -> plus size (title says "Plus Size")
  soft: [waterproof, sealed-seams, hooded, lightweight]
  declined: []
  discard: []
  ceiling: rank-8                 # bucket 49, target 8th by popularity
  notes: |
    "keep me out of the water when it's pouring" -> waterproof. Target's first
    feature is literally "waterproof material", so this bridge is short.
    "designs to prevent water coming in at seams and zippers" -> the target says
    "front zip and wind buckle design, drawstring hoodie and drawstring hem".
```

---

## 07. `B074N8JH9G` — stratum B jargon in features (vocab gap)

**Skysole Boys Fleece Clog Slipper with Rugged Outsole**

- category path: `Clothing, Shoes & Jewelry > Boys > Shoes > Slippers`
- coarse bucket: `Shoes Slippers` (538 siblings)
- brand: `Skysole` | department: `boys` | price: `14.99` | ratings: 1,150
- features: ["Manmade", "Imported", "Foam sole", "SAFETY FIRST: The rugged outsole provides the right amount of traction to the slippers, so he can walk or run without slipping or sliding", "ALL YEAR ROUND: These fleece clogs provide your child some warmth during chilly seasons and enough insulation on sunny days, making it an all-weather pair of slip-on"]
- details: {"Department": "boys", "Date First Available": "August 8, 2017"}

> **What this tests:** Vocabulary gap. Features use jargon the user will never type. Tests synonym bridging (PMI / encoder).

```yaml
utterance: "i'm a father shopping for my 9 year old son. looking for a pair of comfy slippers with like the furry design on the inside of the shoe so that it keeps his little feet warm during the later months and hopefully something with some good grip so my son stays safe."            # <- write the human sentence here
expected_parse:
  hard:
    category: slippers
    department: boys              # !! NOT mens - see notes
    size: kids                    # "9 year old"
  soft: [fleece-lined, warm, grippy, comfortable, non-slip]
  declined: []
  discard: ["so my son stays safe"]
  ceiling: needs-constraints      # pop-rank 32/538 - popularity alone FAILS here
  notes: |
    SPEAKER != WEARER. "i'm a father shopping for my 9 year old son" - the buyer
    is male, the department is boys. Any parser that extracts gender from the
    first person pronoun gets this backwards. New failure mode; not in my taxonomy.
    Bridges: "furry design on the inside" -> fleece (3.0% of catalog)
             "good grip"                  -> outsole/traction (6.5%)
```

---

## 08. `B07K34RX5J` — stratum B jargon in features (vocab gap)

**Kandinsky Statement Earrings for Women by Spirit Hoops, Fabric, Lightweight Drop and Dangle Stainless Steel Hoop Earrings for Women Fashion, Artsy**

- category path: `Clothing, Shoes & Jewelry > Women > Jewelry > Earrings > Hoop`
- coarse bucket: `Earrings Hoop` (200 siblings)
- brand: `Spirit Hoops` | department: `Womens` | price: `— (absent)` | ratings: 871
- features: ["Spandex", "Made in USA and Imported", "Fashion jewelry: Beautiful fabric earrings for women featuring a painting by Kandinsky are wearable art.", "COMFORTABLE: Lightweight dangle earrings, hoops measure approximately 2 inches in diameter. Drop and Dangle hoops with soft fabric covers handmade with care in the U.S.A!", "Great for women of all ages."]
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "1.97 x 1.97 x 0.08 inches; 0.5 Ounces", "Department": "Womens", "Date First Available": "October 29, 2018", "Manufacturer": "Spirit Hoops"}

> **What this tests:** Vocabulary gap. Features use jargon the user will never type. Tests synonym bridging (PMI / encoder).

```yaml
utterance: "i'm looking for a pair of earrings that would catch people's eyes. something artful but also bold and shows taste. maybe something like a hoop earrings."            # <- write the human sentence here
expected_parse:
  hard:
    category: hoop-earrings
    department: null              # never stated (target is Womens) - do not guess
  soft: [statement, bold, eye-catching, artistic, lightweight]
  declined: []
  discard: []
  ceiling: rank-8                 # bucket 200
  notes: |
    "artful / shows taste" -> the target features say "featuring a painting by
    Kandinsky are wearable art". The bridge WORKS but is weak: "art" appears in
    30.9% of the catalog, so it carries almost no IDF. The discriminative token
    is "kandinsky", which no user would ever type. Expect soft-match only.
```

---

## 09. `B07RWZDSM1` — stratum B jargon in features (vocab gap)

**meilun Women's Maxi Bandage Dress Fishtail Bodycon Formal Evening Dresses**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Dresses > Formal`
- coarse bucket: `Dresses Formal` (131 siblings)
- brand: `meilun` | department: `womens` | price: `— (absent)` | ratings: 367
- features: ["90%polyester,9%nylon,1%spandex", "\u8fdb\u53e3", "Pull On closure", "Model Wear size S (tight like glove); Height:5'7/174cm; Bust:(34C)34.3''/87cm; Waist:26''/66cm; Hip:34.6''/88cm", "Maxi dress.She is cut sexy V Neck with a simple design and a floor sweeping fishtail skirt."]
- details: {"Is Discontinued By Manufacturer": "No", "Package Dimensions": "17.63 x 14.48 x 1.89 inches; 2.29 Pounds", "Department": "womens", "Date First Available": "September 21, 2016"}

> **What this tests:** Vocabulary gap. Features use jargon the user will never type. Tests synonym bridging (PMI / encoder).

```yaml
utterance: "i need something that looks classy bold and provocative for a night event. i'm a woman. thinking of liek a nightgown or like a dress to wear to a party or event. in terms of colour preferences maybe something brighter. i need it to show off my figure as well."            # <- write the human sentence here
expected_parse:
  hard:
    category: evening-dress       # !! NOT nightgown - see notes
    department: womens            # stated
  soft: [bodycon, figure-hugging, bold, classy, formal, bright-colour]
  declined: []
  discard: []
  ceiling: needs-constraints      # pop-rank 19/131
  notes: |
    CATEGORY TRAP. The user says "nightgown", which retrieves 296 products of
    which 178 are in "Sleep & Lounge Nightgowns & Sleepshirts". The target is in
    "Dresses Formal" and never uses the word. The parser MUST let the later
    clause ("dress to wear to a party or event") override the earlier wrong noun.
    Taking the user's category word literally sends you to sleepwear.
    Best bridge: "show off my figure" -> bodycon, present in target, 0.9% of
    catalog. That single term is worth more than everything else combined.
    "something brighter" is a colour preference -> SOFT only (colour fails the
    exclusivity test: 15% of products list several colours).
```

---

## 10. `B085S67VBB` — stratum B jargon in features (vocab gap)

**SOJOS Small Retro Square Sunglasses with Rivets Flat Lens Sunnies DAYTIME SJ2114**

- category path: `Clothing, Shoes & Jewelry > Women > Accessories > Sunglasses & Eyewear Accessories > Sunglasses`
- coarse bucket: `Sunglasses & Eyewear Accessories Sunglasses` (522 siblings)
- brand: `SOJOS` | department: `Womens` | price: `— (absent)` | ratings: 427
- features: ["Plastic frame", "Polycarbonate lens", "Non-Polarized", "UV Protection Coating coating", "Lens width: 57 millimeters"]
- details: {"Package Dimensions": "6.2 x 2.4 x 1.6 inches; 2.4 Ounces", "Item model number": "SJ2114C1", "Department": "Womens", "Date First Available": "March 11, 2020", "Manufacturer": "SOJOS"}

> **What this tests:** Vocabulary gap. Features use jargon the user will never type. Tests synonym bridging (PMI / encoder).

```yaml
utterance: "i'm looking for a pair of sunnies that look classy and fits with the throwback look."            # <- write the human sentence here
expected_parse:
  hard:
    category: sunglasses
    department: null              # never stated (target is Womens)
  soft: [retro, vintage, classy, square-frame]
  declined: []
  discard: []
  ceiling: rank-19                # bucket 522, thin utterance
  notes: |
    Shortest utterance in the set and it still works, because both content words
    are rare: "throwback" -> retro (1.8% of catalog) and "sunnies" appears
    VERBATIM in the target title (<0.05% of catalog - near-maximum IDF).
    Lucky lexical hit: the slang the user reached for is in the product title.
    Good control case - shows a two-word query can beat a paragraph when the
    words are rare.
```

---

## 11. `B08N5LWFFC` — stratum B jargon in features (vocab gap)

**J. Adams Cyprus Booties for Women - Strappy Cutout Peep Toe Chunky Heeled Boots**

- category path: `Clothing, Shoes & Jewelry > Women > Shoes`
- coarse bucket: `Women Shoes` (765 siblings)
- brand: `J. Adams` | department: `womens` | price: `— (absent)` | ratings: 159
- features: ["100% Synthetic", "made with synthetic vegan materials sole", "Heel measures approximately 2.25 inches\"", "CHIC & STYLISH: Cyprus by J. Adams is a new take on the trendy ankle bootie. Featuring a flirty peep-toe, a thick mid heel, and a soft faux leather upper with cutouts on the sides, decorative straps, and metal studs.", "INSPIRED ATTIRE: Elevate your style with a pair of Cyprus ankle boots. 
- details: {"Is Discontinued By Manufacturer": "No", "Package Dimensions": "11.34 x 10.2 x 4.09 inches; 1.5 Pounds", "Item model number": "cypruslttaupeimsu-8", "Department": "womens", "Date First Available": "November 10, 2020"}

> **What this tests:** Vocabulary gap. Features use jargon the user will never type. Tests synonym bridging (PMI / encoder).

```yaml
utterance: "i'm looking for one of those pair of boots with high heels and with the toe sticking out a little bit like for that little bit of flair and sexiness (i'm a woman in case you didn't know)"            # <- write the human sentence here
expected_parse:
  hard:
    category: ankle-boots         # "booties"
    department: womens            # stated, parenthetically
  soft: [peep-toe, chunky-heel, high-heel, strappy, flirty]
  declined: []
  discard: []
  ceiling: rank-8                 # bucket 765 but target is 8th by popularity
  notes: |
    Best bridge in the whole set: "toe sticking out a little bit" -> peep toe,
    present in the target and in only 0.3% of the catalog. That one inference
    nearly identifies the product by itself.
    Do NOT discard "(i'm a woman in case you didn't know)" - the aside is
    parenthetical but it is the ONLY department signal in the sentence.
```

---

## 12. `B006H1I85K` — stratum C boilerplate-only features

**ASICS Womens GEL-Blurr33 TR**

- category path: `Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Fitness & Cross-Training`
- coarse bucket: `Athletic Fitness & Cross-Training` (136 siblings)
- brand: `ASICS` | department: `Womens` | price: `— (absent)` | ratings: 377
- features: ["100% Synthetic and mesh", "Rubber sole"]
- details: {"Package Dimensions": "13.1 x 7.8 x 4.6 inches; 1.3 Pounds", "Item model number": "GEL-BLUR33 TR-W", "Department": "Womens", "Date First Available": "March 4, 2012", "Manufacturer": "ASICS"}

> **What this tests:** Boilerplate-only. Every feature is generic - lexical matching has nothing to grip. Tests priors + category.

```yaml
utterance: "i want a pair of shoes that are good for doing a variety of different sports, including racket sports and lifting weights. i am a woman. i think i prefer brands like asics."            # <- write the human sentence here
expected_parse:
  hard:
    category: cross-training-shoes
    department: womens            # stated
    brand: asics                  # stated, hedged ("i think i prefer")
  soft: [versatile, racquet-sports, weightlifting]
  declined: []
  discard: []
  ceiling: achievable             # pop-rank 14/136, but brand+dept narrows to 21 ASICS
  notes: |
    CONTROLLED PAIR with case 13 - see that case's notes. Features here are pure
    boilerplate (["100% Synthetic and mesh", "Rubber sole"]), so brand + category
    + department carry the entire load.
```

---

## 13. `B006H32MDC` — stratum C boilerplate-only features

**ASICS Mens Cael V5.0**

- category path: `Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Tennis & Racquet Sports`
- coarse bucket: `Athletic Tennis & Racquet Sports` (74 siblings)
- brand: `ASICS` | department: `Mens` | price: `— (absent)` | ratings: 329
- features: ["100% Leather", "Rubber sole"]
- details: {"Product Dimensions": "1 x 1 x 1 inches; 0.96 Ounces", "Item model number": "CAEL V5.0-M", "Department": "Mens", "Date First Available": "March 4, 2012", "Manufacturer": "Zappos - FBZ setup"}

> **What this tests:** Boilerplate-only. Every feature is generic - lexical matching has nothing to grip. Tests priors + category.

```yaml
utterance: "i want a pair of high quality, probably like leather shoes that are good for doing a variety of different sports, including racket sports and lifting weights. i am a man. i think i prefer brands like asics"            # <- write the human sentence here
expected_parse:
  hard:
    category: tennis-shoes        # racquet sports - NOT cross-training (cf. case 12)
    department: mens              # stated
    brand: asics                  # stated
    material: leather             # hedged ("probably like leather") - and CORRECT
  soft: [versatile, weightlifting, high-quality]
  declined: []
  discard: []
  ceiling: achievable             # pop-rank 8/74, 12 ASICS in bucket
  notes: |
    CONTROLLED PAIR with case 12. Near-identical utterances; the ONLY difference
    is gender ("i am a man" vs "i am a woman"), and the correct categories are
    different buckets. A parser mapping use-case -> category without conditioning
    on department must get one of the two wrong.
    Material note: "probably like leather" is a HEDGED guess and it happens to be
    right (target features say "100% Leather"). Enforcement = min(user demand,
    catalog capability): hedged demand -> boost, not filter. Getting it right by
    filtering here would be luck, not correctness.
```

---

## 14. `B01BN0MLE8` — stratum C boilerplate-only features

**Wax Denim Women's Juniors Distressed Slim Fit Stretchy Skinny Jeans**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Jeans`
- coarse bucket: `Women Jeans` (319 siblings)
- brand: `Wax` | department: `Womens` | price: `— (absent)` | ratings: 3,311
- features: ["Button closure", "Machine Wash"]
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "1 x 0.1 x 3 inches; 14.4 Ounces", "Item model number": "93300", "Department": "Womens", "Date First Available": "February 10, 2016"}

> **What this tests:** Boilerplate-only. Every feature is generic - lexical matching has nothing to grip. Tests priors + category.

```yaml
utterance: "i want a good pair of jeans that show enough figure like slim fitting but is not too out there. comfortable, maybe stretchy material? something that is just easy to wash, standard pair of denim jeans."            # <- write the human sentence here
expected_parse:
  hard:
    category: jeans
    department: womens
    material: denim               # stated explicitly
  soft: [slim-fit, stretchy, machine-washable, comfortable, understated]
  declined: []
  discard: []
  ceiling: rank-12                # bucket 319
  notes: |
    Stratum C working as intended: features are only ["Button closure",
    "Machine Wash"]. Every discriminative word the user reached for
    (slim, stretchy, skinny, distressed) lives in the TITLE, not the features.
    Evidence that title weighting matters most exactly when features are empty.
    "not too out there" is a negated vibe -> soft, near-zero IDF, harmless.
```

---

## 15. `B079P4GPY4` — stratum C boilerplate-only features

**NINE WEST Women's Slip on Pump**

- category path: `Clothing, Shoes & Jewelry > Women > Shoes > Pumps`
- coarse bucket: `Shoes Pumps` (630 siblings)
- brand: `NINE WEST` | department: `Womens` | price: `— (absent)` | ratings: 434
- features: ["Rubber sole", "Heel measures approximately 3.5 inches\"", "Slip on pump"]
- details: {"Is Discontinued By Manufacturer": "No", "Package Dimensions": "10.39 x 7.01 x 4.33 inches; 8 Ounces", "Item model number": "NW7FILLED9X", "Department": "Womens", "Date First Available": "March 6, 2018", "Manufacturer": "Nine West"}

> **What this tests:** Boilerplate-only. Every feature is generic - lexical matching has nothing to grip. Tests priors + category.

```yaml
utterance: "looking for a pair of high heels to wear to the office, looks more plain like maybe something black and with a skinny stem"            # <- write the human sentence here
expected_parse:
  hard:
    category: pumps
    department: womens
  soft: [black, stiletto, plain, office-wear]
  declined: []
  discard: []
  ceiling: NOT-IDENTIFIABLE
  notes: |
    EXPECTED TO FAIL - keep it. Of six content words, only "pump" and "heel"
    appear anywhere in the target:
        stiletto: absent | black: absent | skinny: absent
        plain:    absent | office: absent
    "skinny stem" -> stiletto is a lovely human phrase with no landing site: the
    product never says stiletto. Colour is absent from the listing entirely, so
    "black" cannot match or contradict. pop-rank 32/630.
    Second unsolvable case in the set (with 05). Both are needed.
```

---

## 16. `B07XC9CGZQ` — stratum C boilerplate-only features

**Breast Lift Tape - 3 Pairs w/Nipple Covers - Boob Tape - Boobtape Caramel**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lingerie > Accessories > Breast Petals`
- coarse bucket: `Accessories Breast Petals` (5 siblings)
- brand: `TwiinsBra` | department: `—` | price: `— (absent)` | ratings: 590
- features: ["Pull-On closure", "Hand Wash"]
- details: {"Brand": "TwiinsBra", "Pattern": "Solid", "Product Care Instructions": "Hand Wash", "Unit Count": "3 Count", "Is Discontinued By Manufacturer": "No", "Package Dimensions": "8.62 x 7.76 x 0.47 inches; 2.89 Ounces"}

> **What this tests:** Boilerplate-only. Every feature is generic - lexical matching has nothing to grip. Tests priors + category.

```yaml
utterance: "i'm going to wear a strapless top which might be quite revealing so i need something that can cover my nipples and accentuate my boobs a bit as well. and hopefully something like skin coloured so if any wardrobe malfunctions were to happen it wouldn't be too obvious"            # <- write the human sentence here
expected_parse:
  hard:
    category: breast-petals       # nipple covers / fashion tape
    department: womens
  soft: [skin-toned, lift, invisible, strapless-compatible]
  declined: []
  discard: []
  ceiling: rank-2                 # bucket has only 5 products - category IS the answer
  notes: |
    Inverse of case 15: almost nothing rides on the soft terms, everything rides
    on resolving the category. Get to "Accessories Breast Petals" (5 products)
    and you win at rank 2. But there is ZERO shared vocabulary between "something
    that can cover my nipples" and the category name - pure inference.
    "skin coloured" -> the target's shade is "Caramel". The words skin/nude/beige
    are all absent. Requires knowing caramel is a skin-tone name.
```

---

## 17. `B016HCI5DS` — stratum D model-code / spec-driven

**Lian LifeStyle Children 3 or 6 Pairs Wool Crew Boot Socks HRL1801 Size 0M-24M**

- category path: `Clothing, Shoes & Jewelry > Baby > Baby Boys > Accessories > Socks`
- coarse bucket: `Accessories Socks` (48 siblings)
- brand: `Lian LifeStyle` | department: `Baby-girls` | price: `— (absent)` | ratings: 144
- features: ["Wool Blend", "Imported", "Wool", "Machine Wash", "THE GREATEST COMFORT: Exceptional children socks particularly engineered to make walking and playing enjoyable and relaxing. Their superior wool fibers blend ensures that your little man experiences only premium warmth. Reduce foot fatigue and ankle swelling while spoiling tiny toes."]
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "7 x 5 x 1.9 inches; 5 Ounces", "Item model number": "Cotton", "Department": "Baby-girls", "Date First Available": "June 2, 2015"}

> **What this tests:** Spec/model-code. Distinctive alphanumerics that embeddings destroy and exact matching nails.

```yaml
utterance: "I'm looking for wool socks for my children. something easy to wash would be preferred."            # <- write the human sentence here
expected_parse:
  hard:
    category: socks
    material: wool                # stated, and present in target features
    department: null              # !! see notes - do NOT extract
  soft: [machine-washable, warm, childrens, comfortable]
  declined: []
  discard: []
  ceiling: rank-10                # bucket 48
  notes: |
    CATALOG CONTRADICTS ITSELF. details.Department = "Baby-girls", but the
    product's own description says "your little man" and "boy", and the word
    "girl" never appears in the text. Filtering on department is a coin flip
    against the catalog's own data.
    The user said "my children" with no gender - which is the correct behaviour
    and must NOT be resolved to a department. Leave it null.
    Clean bridges otherwise: "wool" -> "Wool Blend"/"Wool" (present),
    "easy to wash" -> "Machine Wash" (present).
```

---

## 18. `B07SMPKSZH` — stratum D model-code / spec-driven

**KOOFIN GEAR Performance Fishing Hoodie UPF50 Sunblock Shirt Outdoor Quick-Dry Athletic Sweatshirt**

- category path: `Clothing, Shoes & Jewelry > Men > Clothing > Shirts`
- coarse bucket: `Men Shirts` (32 siblings)
- brand: `KOOFIN GEAR` | department: `mens` | price: `— (absent)` | ratings: 411
- features: ["Pull On closure", "Machine Wash", "UPF50 sun protection; dry fit stretch-flex fabric; Flatlock stitching; No shoulder seams; Unbelievably soft and lightweight. It is a good basic shirt for workouts or for casual wear.", "Moisture-wicking; breathes well and dries quickly; keeps you dry and comfortable.", "All-Over Dye Sublimation Print; Will Never Crack or Fade; Machine Washable."]
- details: {"Package Dimensions": "11.77 x 9.96 x 1.93 inches; 9.52 Ounces", "Item model number": "FHH006", "Department": "mens", "Date First Available": "June 14, 2019", "Manufacturer": "KOOFIN"}

> **What this tests:** Spec/model-code. Distinctive alphanumerics that embeddings destroy and exact matching nails.

```yaml
utterance: "need one of those long sleeve shirts for fishing, the kind that blocks the sun. upf something? 50 i think. gotta be quick drying too cause i sweat a lot out on the boat"
expected_parse:
  hard:
    category: mens-shirts
    department: mens
  soft: [upf-50, sun-protection, quick-dry, moisture-wicking, long-sleeve, fishing]
  declined: []
  discard: ["cause i sweat a lot out on the boat"]
  ceiling: rank-2                 # bucket 32, poprank 2
  notes: |
    HALF-REMEMBERED SPEC. "upf something? 50 i think" - the user hedges on the
    exact figure. "UPF50" appears verbatim in the target title. Recovering a
    hedged alphanumeric spec is worth more than every vibe word combined.
```

---

## 19. `B08K8HVQN4` — stratum D model-code / spec-driven

**BGOWATU Men's Golf Polo Shirts 3-Button Lightweight UPF 50+ Long Sleeve Stretch Athletic Casual T-Shirt**

- category path: `Clothing, Shoes & Jewelry > Men > Clothing > Active > Active Shirts & Tees`
- coarse bucket: `Active Active Shirts & Tees` (263 siblings)
- brand: `BGOWATU` | department: `—` | price: `— (absent)` | ratings: 80
- features: ["Fabric: 92% Polyester & 8% Elastans", "Button closure", "Machine Wash", "Lightweight, Breathable and Quick-Dry fabric is good quality with a thin fleece lining, keeping you warm and dry in outdoor activities.", "Wrinkle resistant, stretch material with no tag collar design minimize friction and increase comfort."]
- details: {"Brand": "BGOWATU", "Color": "Green", "Fit Type": "Athletic Fit", "Style": "Casual", "Neck Style": "Collared Neck", "Age Range (Description)": "Adult"}

> **What this tests:** Spec/model-code. Distinctive alphanumerics that embeddings destroy and exact matching nails.

```yaml
utterance: "looking for a golf shirt for my dad, long sleeve, the sun protective type. he likes the ones with the buttons at the collar. stretchy would be good, he's not exactly slim these days"
expected_parse:
  hard:
    category: active-shirts
    department: mens              # "for my dad" - NOT the speaker's own gender
  soft: [golf, upf-50, long-sleeve, button-collar, stretch, lightweight]
  declined: []
  discard: ["he's not exactly slim these days"]
  ceiling: needs-constraints      # poprank 31/263 - popularity alone FAILS
  notes: |
    SPEAKER != WEARER again (cf. case 07). Second-hardest case in stratum D:
    only 80 ratings, so the popularity prior actively works against it.
    "buttons at the collar" -> the target is a "3-Button" polo. Everything rides
    on the spec terms; the vibe words are worthless here.
```

---

## 20. `B0BNP1RZ2W` — stratum D model-code / spec-driven

**PAVOI 14K Gold Plated Lightweight Chunky Open Hoops | Gold Hoop Earrings for Women**

- category path: `Clothing, Shoes & Jewelry > Women > Jewelry > Earrings > Hoop`
- coarse bucket: `Earrings Hoop` (200 siblings)
- brand: `PAVOI` | department: `womens` | price: `59.99` | ratings: 43,974
- features: ["\u272618K GOLD VERMEIL\u2726 A premium offering of our best selling styles, our vermeil jewelry is made with a solid s925 sterling silver base and plated in 18K gold that is\u00a010x thicker than the industry standard 'plated gold'. Vermeil is a very durable, hypoallergenic material, made to last.", "PAVOI 4.5mm Thick 40mm Diameter Yellow Gold Earrings for Women", "\u2726 60-DAY GUARANTEE \u2726
- details: {"Department": "womens", "Date First Available": "January 2, 2019", "Manufacturer": "PAVOI"}

> **What this tests:** Spec/model-code. Distinctive alphanumerics that embeddings destroy and exact matching nails.

```yaml
utterance: "want gold hoop earrings, the chunky kind not the thin dainty ones. not real gold obviously, plated is fine. budget is like 60 bucks max"
expected_parse:
  hard:
    category: hoop-earrings
    department: womens
    price: {max: 60}              # PRICE CASE - target is $59.99, just inside
  soft: [chunky, gold-plated, lightweight]
  negated: [thin, dainty, solid-gold]
  declined: []
  discard: []
  ceiling: rank-1                 # poprank 1/200, 43,974 ratings
  notes: |
    PRICE CASE, and the only one where price is genuinely decisive: target is
    $59.99 against a stated $60 ceiling. One dollar of slack.
    Reminder that price has 21.1% catalog coverage - filtering on it must never
    exclude products with a null price, or you drop 79% of the catalog.
    "not real gold obviously, plated is fine" is a negation IMMEDIATELY followed
    by its own relaxation. Parsers that only catch the "not" get it backwards.
```

---

## 21. `B083WGBK79` — stratum E sparse text (hard mode)

**SANMIO Girl Clothes Outfits (1-6T), Toddler Baby Girls Clothes Camouflage Printed Outfits for Girls Sweatshirt Clothing Set**

- category path: `Clothing, Shoes & Jewelry > Baby > Baby Girls > Clothing > Clothing Sets > Pant Sets`
- coarse bucket: `Clothing Sets Pant Sets` (203 siblings)
- brand: `SANMIO` | department: `Baby-girls` | price: `— (absent)` | ratings: 853
- features: []
- details: {"Package Dimensions": "13.27 x 9.33 x 0.63 inches; 3.6 Ounces", "Department": "Baby-girls", "Date First Available": "February 10, 2020"}

> **What this tests:** Sparse text. Almost no features, no description. Tests title-only + category + popularity.

```yaml
utterance: "my daughter is turning 2 and i want to get her one of those matching outfit sets, like top and pants together. she likes the army print thing"
expected_parse:
  hard:
    category: clothing-sets
    department: baby-girls        # "my daughter", age 2
    size: toddler
  soft: [camouflage, matching-set, two-piece]
  declined: []
  discard: ["my daughter is turning 2"]   # occasion; the AGE is a constraint, the birthday is not
  ceiling: rank-7                 # bucket 203
  notes: |
    Stratum E: the target has ZERO features. Title is the only text.
    Bridge: "army print" -> "camouflage"/"camo" (both present in title, "army"
    absent). Short bridge but it must be made.
    Third speaker != wearer case. Age 2 -> toddler sizing -> Baby-girls.
```

---

## 22. `B08DL4SL2M` — stratum E sparse text (hard mode)

**Romwe Women's Elegant Short Puff Sleeve Rib Knit V-Neck Basic Slim Fit T-Shirt Crop Tops Royal Blue Medium**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts`
- coarse bucket: `Tees & Blouses T-Shirts` (680 siblings)
- brand: `Romwe` | department: `Womens` | price: `— (absent)` | ratings: 1,939
- features: ["Polyester,Cotton,Spandex", "Machine Wash"]
- details: {"Product Dimensions": "5.91 x 5.91 x 1.97 inches; 5.19 Ounces", "Item model number": "46-swTS07200717608-M", "Department": "Womens", "Date First Available": "August 12, 2020"}

> **What this tests:** Sparse text. Almost no features, no description. Tests title-only + category + popularity.

```yaml
utterance: "looking for a plain fitted short sleeve top, cropped, with a v neckline. ribbed material maybe? just something basic i can layer under stuff"
expected_parse:
  hard:
    category: t-shirts
    department: womens            # weakly inferred from "cropped" - not stated
  soft: [crop-top, v-neck, ribbed, slim-fit, short-sleeve, basic, layering]
  declined: []
  discard: []
  ceiling: needs-constraints      # poprank 20/680
  notes: |
    Every soft term here appears in the target TITLE ("Short Puff Sleeve Rib
    Knit V-Neck Basic Slim Fit T-Shirt Crop Tops"), and the features are only
    ["Polyester,Cotton,Spandex", "Machine Wash"].
    Same lesson as case 14: when features are boilerplate, the title carries
    everything. Argues for title weighting over feature weighting.
```

---

## 23. `B08GKV2M6R` — stratum E sparse text (hard mode)

**Jileon Ankle Rain Boots Wide Width Fit | Specially Designed For Ladies with Wide Feet & Calves | Wide Calf Rain Boots for Women | Rain Boots for Plus Size Women**

- category path: `Clothing, Shoes & Jewelry > Boot Shop > Women > Outdoor & Work > Rain`
- coarse bucket: `Outdoor & Work Rain` (109 siblings)
- brand: `Jileon` | department: `womens` | price: `— (absent)` | ratings: 1,356
- features: ["Rubber sole"]
- details: {"Package Dimensions": "12.2 x 10.4 x 4.6 inches; 2.9 Pounds", "Department": "womens", "Date First Available": "December 12, 2020"}

> **What this tests:** Sparse text. Almost no features, no description. Tests title-only + category + popularity.

```yaml
utterance: "i have really wide feet and wide calves and normal rain boots never fit me. need ankle height ones, women's. honestly the wide fit is the whole thing, everything else i don't care about"
expected_parse:
  hard:
    category: rain-boots
    department: womens
    size: wide-width
  soft: [wide-calf, ankle-height]
  declined: [colour, material, brand, price]   # "everything else i don't care about"
  discard: ["normal rain boots never fit me"]
  ceiling: rank-7                 # bucket 109
  notes: |
    VERY SPECIFIC + BLANKET DECLINE. Cleanest bridges in the whole set - wide,
    calf, calves, ankle ALL present in the target title.
    The blanket decline is the interesting part: one clause declines four
    attributes at once. A slot-filling parser has no natural way to express
    "everything else", and treating it as silence loses the information that
    the user has explicitly released those constraints.
```

---

## 24. `B08VDM4G8B` — stratum E sparse text (hard mode)

**1950s Pink Satin Jacket with Neck Scarf Grils Women Danny Halloween Costume Fancy Dress**

- category path: `Clothing, Shoes & Jewelry > Costumes & Accessories > Women > Costumes & Cosplay Apparel > Robes, Capes & Jackets`
- coarse bucket: `Robes Capes & Jackets` (26 siblings)
- brand: `CISSTEC` | department: `womens` | price: `22.99` | ratings: 2,073
- features: []
- details: {"Department": "womens", "Date First Available": "December 9, 2020"}

> **What this tests:** Sparse text. Almost no features, no description. Tests title-only + category + popularity.

```yaml
utterance: "halloween's coming up and i'm going as sandy from grease, need that pink jacket thing with the scarf. nothing expensive, like $25 tops"
expected_parse:
  hard:
    category: costume-jacket
    department: womens
    price: {max: 25}              # PRICE CASE - target is $22.99
  soft: [pink, satin, neck-scarf, 1950s, halloween-costume]
  declined: []
  discard: ["halloween's coming up"]
  ceiling: rank-2                 # bucket 26, poprank 2
  notes: |
    CULTURAL REFERENCE, PARTIALLY BROKEN. "sandy from grease" requires knowing
    the film AND the character's costume. Worse: "sandy" and "grease" are BOTH
    absent from the target, which instead indexes "Danny" - the male lead.
    The reference resolves to the wrong character in the catalog's own text.
    Target has ZERO features; title-only.
    Second price case, and here price is a loose bound ($22.99 vs $25).
```

---

## 25. `B07KCFS4VC` — stratum F crowded bucket + priced

**Columbia Men's Thistletown Park Crew**

- category path: `Clothing, Shoes & Jewelry > Men > Clothing > Shirts > T-Shirts`
- coarse bucket: `Shirts T-Shirts` (1354 siblings)
- brand: `Columbia` | department: `mens` | price: `27.99` | ratings: 5,531
- features: ["67% Polyester, 33% Cotton", "Imported", "No Closure closure", "Machine Wash", "ADVANCED TECHNOLOGY: Columbia Men's Thistletown Park Short Sleeve Crew Shirt features our signature UPF 15 fabric that helps to blocks harmful UVA and UVB rays as well as wicking technology to help keep you dry and cool on the trail."]
- details: {"Item model number": "1441396", "Department": "mens", "Date First Available": "January 13, 2015", "Manufacturer": "Columbia"}

> **What this tests:** Crowded bucket (400+ siblings) with a price. Tests tie-breaking and price handling on a 21%-coverage field.

```yaml
utterance: "need a couple of plain tees for my husband, he's outdoorsy and wears them hiking. under 30 dollars each. he hates anything that feels like plastic against his skin"
expected_parse:
  hard:
    category: t-shirts
    department: mens              # "my husband"
    price: {max: 30}              # PRICE CASE - target is $27.99
  soft: [plain, outdoor, breathable, natural-feel]
  negated: []                     # !! see notes - do NOT negate synthetic
  declined: []
  discard: ["he's outdoorsy"]
  ceiling: rank-8                 # bucket 1,354 - largest in the set
  notes: |
    NEGATION TRAP - the most important case in this batch.
    "hates anything that feels like plastic" reads as "not synthetic". The
    target is 67% POLYESTER. Bridge plastic -> polyester, negate it, and you
    delete the correct answer.
    Note the words "plastic" and "synthetic" are both ABSENT from the target, so
    there is no contradiction term to match on either - the exclusion would be
    made entirely on inference. Exactly the case where a negation must stay soft.
    Fourth speaker != wearer case. Third price case.
```

---

## 26. `B08MBM15JB` — stratum F crowded bucket + priced

**Evshine Women's Fuzzy Slippers Cross Band Memory Foam House Slippers Open Toe**

- category path: `Clothing, Shoes & Jewelry > Women > Shoes > Slippers`
- coarse bucket: `Shoes Slippers` (538 siblings)
- brand: `Evshine` | department: `womens` | price: `11.99` | ratings: 4,186
- features: ["Ethylene Vinyl Acetate sole", "FASHION & ELEGANT: Breathable Open-toe along with trendy faux fur design makes these womens slippers stylish and practical. Easily slides in whenever you want your feet relaxed.", "FUZZY HOUSE SLIPPERS: Fuzzy faux fur upper and footbed surrounds your foot in cloud comfort, making your feet cozy. Definitely a good choice for reducing muscle fatigue after a long day 
- details: {"Department": "womens", "Date First Available": "October 31, 2020"}

> **What this tests:** Crowded bucket (400+ siblings) with a price. Tests tie-breaking and price handling on a 21%-coverage field.

```yaml
utterance: "cheap house slippers, the fluffy open toe kind with memory foam in them. don't want to spend more than like 15"
expected_parse:
  hard:
    category: slippers
    department: womens            # weakly inferred, not stated
    price: {max: 15}              # PRICE CASE - target is $11.99
  soft: [fuzzy, faux-fur, open-toe, memory-foam, house-slippers]
  declined: []
  discard: []
  ceiling: rank-12                # bucket 538
  notes: |
    BUDGET END of the price range (target $11.99 vs catalog median $22.88).
    Fourth price case, deliberately at the cheap end - the three others sit at
    $22.99 / $27.99 / $59.99, so the set spans the distribution.
    "cheap" is catalog-relative and must be grounded in the price distribution;
    "more than like 15" is the actual bound. Both appear in one sentence.
```

---

## 27. `B08W4JXR19` — stratum F crowded bucket + priced

**Dokotoo Women's V Neck Lace Crochet Eyelet Tops Short Sleeve Casual Shirts Blouses**

- category path: `Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses & Button-Down Shirts`
- coarse bucket: `Tees & Blouses Blouses & Button-Down Shirts` (681 siblings)
- brand: `Dokotoo` | department: `womens` | price: `29.99` | ratings: 1,566
- features: ["Lace", "Pull On closure", "Dokotoo womens v neck lace shirts is made with high-quality fabric. Soft, lightweight and comfortable to wear", "Feature: V Neck, Solid Color, Short Sleeve, Button Design, Front Crochet Eyelet Panel, Lightweight Fabrication, Loose Fit Lace Top", "You can pair this fashion blouses with variety of coats, jacket, jeans, denim shorts, skirts, jeggings, sneakers or heels to
- details: {"Item model number": "CT5W2516745-P", "Department": "womens", "Date First Available": "February 7, 2021"}

> **What this tests:** Crowded bucket (400+ siblings) with a price. Tests tie-breaking and price handling on a 21%-coverage field.

```yaml
utterance: "want a nice summery blouse, v neck, short sleeves, with some lace or crochet detail at the front. something i could wear to brunch"
expected_parse:
  hard:
    category: blouses
    department: womens
  soft: [v-neck, short-sleeve, lace, crochet, eyelet, summer, casual]
  declined: []
  discard: ["something i could wear to brunch"]
  ceiling: rank-12                # bucket 681
  notes: |
    Cooperative user, well-specified, no traps - deliberate control case.
    Every soft term lands: the target is "V Neck Lace Crochet Eyelet Tops Short
    Sleeve". If the system cannot get this one, the problem is not parsing.
    "wear to brunch" is a use-case with no catalog vocabulary behind it -
    discard or treat as near-zero-weight soft.
```

---

## 28. `B07GXHPWTJ` — stratum G public_set control

**Angel Barcelo Roomy Fashion Hobo Womens Handbags Ladies Purse Satchel Shoulder Bags Tote Washed Leather Bag**

- category path: `Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Totes`
- coarse bucket: `Handbags & Wallets Totes` (184 siblings)
- brand: `Angel Barcelo` | department: `Womens` | price: `42.99` | ratings: 12,633
- features: ["Soft Washed PU Leather with Convenient side Pockets", "Imported", "Adjustable and Removable Shoulder Strap", "Quality Material:High Quality Anti-Scratch PU Leather Hobo Tote Womens Purse Handbag.soft hand feel and durable,Front U-shaped sewing design, reinforced bottom with special hook decoration, two Side pockets make the hobo bag more unique for women daily use.", "Dimension(L*W*H): Size:13.8
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "10 x 10 x 2 inches; 1.2 Pounds", "Item model number": "0044", "Department": "Womens", "Date First Available": "July 23, 2018"}

> **What this tests:** Public-set control. Same product the templated simulator uses - lets you isolate PARSING difficulty from RETRIEVAL difficulty.

```yaml
utterance: "looking for a big shoulder bag, soft leather-looking, roomy enough to fit a laptop. i only buy stuff with loads of good reviews so nothing obscure please"
expected_parse:
  hard:
    category: tote-bags
    department: womens
  soft: [roomy, hobo, soft-pu-leather, shoulder-strap, laptop-capacity]
  quality:                        # RATINGS CASE - the user asks for the prior explicitly
    well_rated: true              # -> average_rating (4.4)
    high_volume: true             # -> rating_number (12,633)
  declined: []
  discard: []
  ceiling: rank-3                 # bucket 184
  notes: |
    RATINGS CASE. "loads of good reviews", "nothing obscure" - the user is
    asking, in words, for the popularity prior that scores 0.713 standalone.
    Maps to rating_number and average_rating, the only 100%-coverage numeric
    fields. No text attribute exists for it.
    Note "leather-looking" not "leather": the target is PU leather. A hard
    material filter on leather would be wrong in a way the user pre-empted.
```

---

## 29. `B07VCYFB5D` — stratum G public_set control

**Baseball Cap Custom Personalized Text Dad Hats for Men & Women Strap Closure**

- category path: `Clothing, Shoes & Jewelry > Novelty & More > Clothing > Novelty > Women > Accessories > Hats & Caps > Baseball Caps`
- coarse bucket: `Hats & Caps Baseball Caps` (385 siblings)
- brand: `Speedy Pros` | department: `mens` | price: `19.99` | ratings: 2,454
- features: ["100% Acrylic", "Hook and Loop closure", "PREMIUM QUALITY: Take your outfit to the next level with our 100% acrylic 6 panels mid-profile structured baseball hat that provides maximum comfort. Fits men and women!", "HOOK & LOOP CLOSURE: Our hats for women and men feature a strap closure in the back letting you easily adjust the size for a perfect fit.", "PRE-CURVED BILL: Our mens hat and womens ha
- details: {"Item model number": "CUST_CAP_001", "Department": "mens", "Date First Available": "April 22, 2018", "Manufacturer": "Speedy Pros"}

> **What this tests:** Public-set control. Same product the templated simulator uses - lets you isolate PARSING difficulty from RETRIEVAL difficulty.

```yaml
utterance: "i want a baseball cap i can put my own text on, custom printed or embroidered, whatever. needs the adjustable strap at the back. around 20 bucks"
expected_parse:
  hard:
    category: baseball-caps
    department: null              # target says mens; user says nothing. Do not guess.
    price: {max: 20}              # target is $19.99
  soft: [custom-text, personalized, strap-closure, adjustable]
  declined: []
  discard: ["custom printed or embroidered, whatever"]  # method is explicitly indifferent
  ceiling: rank-3                 # bucket 385
  notes: |
    VERY SPECIFIC on the one attribute that matters: "custom text" /
    "personalized" is the product's entire reason to exist and it is rare
    vocabulary. Everything else is generic cap language.
    "whatever" after "printed or embroidered" is a mid-sentence decline of a
    distinction the user raised themselves - neither branch is a constraint.
```

---

## 30. `B08513YB2T` — stratum G public_set control

**Crocs Unisex-Adult Classic Clog**

- category path: `Clothing, Shoes & Jewelry > Men > Shoes > Mules & Clogs`
- coarse bucket: `Shoes Mules & Clogs` (283 siblings)
- brand: `Crocs` | department: `unisex-adult` | price: `— (absent)` | ratings: 408,371
- features: ["Made in the USA or Imported", "Ethylene Vinyl Acetate sole", "Shaft measures approximately 8#inches from arch", "Heel measures approximately 0.77\"", "CROCS FOR EVERYONE: With a color and style for every personality, the Classic Clogs are the Crocs women and men need to start a comfort revolution around the world"]
- details: {"Is Discontinued By Manufacturer": "No", "Product Dimensions": "8 x 4 x 7 inches; 9.6 Ounces", "Item model number": "10001", "Department": "unisex-adult", "Date First Available": "September 20, 2007", "Manufacturer": "Crocs"}

> **What this tests:** Public-set control. Same product the templated simulator uses - lets you isolate PARSING difficulty from RETRIEVAL difficulty.

```yaml
utterance: "those foam clog things with the holes in them that everyone wears, you know the ones. unisex ideally. don't really care what colour"
expected_parse:
  hard:
    category: clogs
    department: unisex-adult
  soft: [foam, lightweight, slip-on]
  declined: [colour]
  discard: ["you know the ones"]
  ceiling: rank-1                 # 408,371 ratings - most-rated product in the catalog
  notes: |
    POPULARITY CASE, and a total lexical failure. The user describes Crocs by
    APPEARANCE and never names the brand. Both descriptive words are absent:
        "foam"  -> ABSENT (the target says "croslite", "ethylene")
        "holes" -> ABSENT
    Zero lexical purchase on the two content words. The ONLY thing that finds
    this is the popularity prior - 408,371 ratings, rank 1 in its bucket.
    Best possible argument in the set for keeping the prior as a first-class
    scoring term rather than a tie-breaker.
```

---
