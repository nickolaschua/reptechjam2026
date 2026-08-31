# Phase 6 candidate-selection report

Generated offline from the current official public set and 50,000-row catalogue. Every selected session reuses an existing public buying row; no ASIN was fabricated. The fixed budget tendency is under approximately $120.

## Selected sessions

### u1_stable — stable learner

#### S1: cold_start

- Source: `public_0018` (buying, easy)
- Target: `B07H3T5YGH` — O2TEE Men's Workout Gym Tank Tops Men - Custom Tank Top - Customized & Personalized Tanktops Text
- Category: Clothing, Shoes & Jewelry > Novelty & More > Clothing > Novelty > Men > Shirts > Tanks Tops
- Longitudinal signal: none
- Directive: none
- Verified properties: category=tank top [categories: Tanks Tops]
- Why appropriate: A buying-row shirt task gives a category reference close to the later T-shirt probe without disclosing tested history.

#### S2: establish_p1

- Source: `public_0199` (buying, easy)
- Target: `B089M57PSQ` — Boboking 100% Cotton Little Boys Briefs Soft Dinosaur Truck Toddler Underwear
- Category: Clothing, Shoes & Jewelry > Boys > Clothing > Underwear > Briefs
- Longitudinal signal: P1 first disclosure
- Directive: disclose: Across purchases, I generally prefer breathable or natural materials.
- Verified properties: material=100% cotton [features[2]: 100% Cotton]; comfort=breathable [features[3]: breathable]
- Why appropriate: The target explicitly states 100% cotton and breathable fabric, directly supporting P1.

#### S3: reinforce_p1

- Source: `public_0024` (buying, easy)
- Target: `B076X3JXMW` — Riviera Sun Womens Off Shoulder Embroidered Jumpsuit Romper
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Jumpsuits, Rompers & Overalls > Jumpsuits
- Longitudinal signal: P1 reinforcement
- Directive: reinforce: I generally prefer breathable or natural materials.
- Verified properties: material=100% rayon [features[0]: 100 rayon]; comfort=keeps cool [features[3]: let body heat escape]
- Why appropriate: A different clothing context explicitly describes heat-escaping rayon fabric.

#### S4: establish_p2

- Source: `public_0029` (buying, easy)
- Target: `B01IAKCZEK` — Sanuk Yoga Sling 2 Light Natural 5 B (M)
- Category: Clothing, Shoes & Jewelry > Women > Shoes > Sandals > Flats
- Longitudinal signal: P2 first disclosure
- Directive: disclose: Across purchases, I generally prefer neutral colours.
- Verified properties: color=light natural [details.Color: Light Natural]
- Why appropriate: The catalogue's target-level Color field is Light Natural, a verified neutral.

#### S5: reinforce_p2

- Source: `public_0156` (buying, easy)
- Target: `B0C3KZXV4B` — adidas Alliance II Sackpack, Shadow Navy/Snowglobe/Dash Grey, One Size
- Category: Clothing, Shoes & Jewelry > Luggage & Travel Gear > Gym Bags > Drawstring Bags
- Longitudinal signal: P2 reinforcement
- Directive: reinforce: I generally prefer neutral colours.
- Verified properties: color=shadow navy and dash grey [title: Shadow Navy/Snowglobe/Dash Grey]
- Why appropriate: The exact target title supplies navy and grey colour evidence in another product family.

#### S6: establish_p3

- Source: `public_0026` (buying, easy)
- Target: `B093R14VP1` — ASICS Men's Gel-Venture 6 MX Running Shoes
- Category: Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Running > Trail Running
- Longitudinal signal: P3 first disclosure under approximately $120
- Directive: disclose: Across purchases, I usually try to stay under approximately $120.
- Verified properties: price=63.95 [price: 63.95]
- Why appropriate: The trail-running target has a reliable numeric price well below the fixed $120 threshold.

#### S7: combine_p1_p3

- Source: `public_0163` (buying, easy)
- Target: `B0834T68X3` — DOUSSPRT Womens Walking Shoes Slip on Sock Sneakers Lady Girls Nurse Mesh Air Cushion Platform Loafers Fashion Casual
- Category: Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
- Longitudinal signal: P1 light reinforcement with P3-compatible target
- Directive: reinforce: I generally prefer breathable materials.
- Verified properties: feature=breathable mesh [features[4]: ventilation and breathability]; price=28.89 [price: 28.89]
- Why appropriate: Ventilation is explicit and the numeric price is below the benchmark threshold.

#### S8: combine_p1_p2

- Source: `public_0108` (buying, easy)
- Target: `B01I21CI7G` — Hanes Women's Stretch Jersey Bike Shorts, Women’s Cotton Bike Shorts, Women’s Athletic Shorts, 7" Inseam
- Category: Clothing, Shoes & Jewelry > Sport Specific Clothing > Cycling > Women > Tights, Pants & Shorts > Shorts
- Longitudinal signal: P1 and P2 light reinforcement
- Directive: reinforce: I generally prefer breathable natural fabrics and neutral colours.
- Verified properties: material=cotton blend [features[0]: 54% Cotton]; color=neutral shades [features[7]: assortment of neutral shades]
- Why appropriate: The product family explicitly uses cotton and is offered as an assortment of neutral shades.

#### S9: mature_history

- Source: `public_0088` (buying, easy)
- Target: `B07Z6J5N6Y` — Amazon Essentials Women's Cotton Bikini Brief Underwear (Available in Plus Size), Multipacks
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lingerie > Panties > Bikinis
- Longitudinal signal: none; mature compatible history
- Directive: none
- Verified properties: material=95% cotton [features[0]: 95% Cotton]; price=12.72 [price: 12.72]
- Why appropriate: A 95% cotton, inexpensive ordinary purchase is broadly compatible without another scheduled disclosure.

#### S10: memory_probe

- Source: `public_0178` (buying, easy)
- Target: `B01FWQ8NH8` — Red Hot Chili Peppers Distressed Men's T-Shirt Black
- Category: Clothing, Shoes & Jewelry > Novelty & More > Clothing > Novelty > Men > Shirts > T-Shirts
- Longitudinal signal: same S10 used for no-history/full-history probe under approximately $120
- Directive: none
- Verified properties: material=100% cotton [features[0]: 100% Cotton]; color=black [title: T-Shirt Black]; price=17.99 [price: 17.99]
- Why appropriate: A shirt task close to S1 has verified cotton, black colour, and numeric price while allowing an underspecified current request.

### u2_override — query override

#### S1: establish_p1

- Source: `public_0149` (buying, easy)
- Target: `B07CBYYHTL` — SDIYABOLO Small Black Sling Crossbody Backpack Shoulder Bag for Men Women Vintage PU Leather Sling Backpack Cycling
- Category: Clothing, Shoes & Jewelry > Luggage & Travel Gear > Backpacks > Casual Daypacks
- Longitudinal signal: P1 first disclosure
- Directive: disclose: Across purchases, I generally prefer dark or neutral colours.
- Verified properties: color=black [title: Small Black Sling]
- Why appropriate: The exact public target title identifies a black backpack.

#### S2: reinforce_p1

- Source: `public_0160` (buying, easy)
- Target: `B01AAANF2Y` — Amazon Basics 4 Piece Packing Travel Organizer Cubes Set - Slim, Black
- Category: Clothing, Shoes & Jewelry > Luggage & Travel Gear > Travel Accessories > Packing Organizers
- Longitudinal signal: P1 reinforcement
- Directive: reinforce: I generally prefer dark or neutral colours.
- Verified properties: color=black [details.Color: Black]
- Why appropriate: The catalogue gives target-level Color=Black.

#### S3: establish_p2

- Source: `public_0028` (buying, easy)
- Target: `B0B9ZYDDZ1` — Buffway Slim Minimalist Front Pocket RFID Blocking Leather Wallets for Men Women
- Category: Clothing, Shoes & Jewelry > Men > Accessories > Wallets, Card Cases & Money Organizers > Wallets
- Longitudinal signal: P2 first disclosure
- Directive: disclose: Across purchases, I generally prefer understated or minimal styling.
- Verified properties: style=minimalist [details.Style: Minimalist]; color=lichee black [details.Color: Lichee Black]
- Why appropriate: Target-level Style=Minimalist and Color=Lichee Black are unusually strong catalogue evidence.

#### S4: reinforce_p2

- Source: `public_0054` (buying, easy)
- Target: `B08PP1ZJQ5` — Verdusa Women's Basic Casual Long Sleeve Round Neck Crop Top Pullover Sweatshirt
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Fashion Hoodies & Sweatshirts
- Longitudinal signal: P2 reinforcement
- Directive: reinforce: I generally prefer understated or minimal styling.
- Verified properties: style=basic casual [title: Basic Casual]
- Why appropriate: The target is explicitly described as basic casual styling.

#### S5: establish_p3

- Source: `public_0042` (buying, easy)
- Target: `B01LWOGORL` — Timex Men's Classic Digital Watch
- Category: Clothing, Shoes & Jewelry > Men > Watches > Wrist Watches
- Longitudinal signal: P3 first disclosure under approximately $120
- Directive: disclose: Across purchases, I usually try to stay under approximately $120.
- Verified properties: price=33.16 [price: 33.16]
- Why appropriate: The classic watch has a reliable $33.16 catalogue price.

#### S6: strong_dark_minimal

- Source: `public_0028` (buying, easy)
- Target: `B0B9ZYDDZ1` — Buffway Slim Minimalist Front Pocket RFID Blocking Leather Wallets for Men Women
- Category: Clothing, Shoes & Jewelry > Men > Accessories > Wallets, Card Cases & Money Organizers > Wallets
- Longitudinal signal: strong P1/P2 reinforcement with price-compatible target
- Directive: reinforce: I strongly prefer dark colours and understated minimal styling.
- Verified properties: style=minimalist [details.Style: Minimalist]; price=15.99 [price: 15.99]
- Why appropriate: Reusing the exceptionally well-audited black minimalist wallet strengthens the controlled history.

#### S7: reinforce_p1_p2

- Source: `public_0152` (buying, easy)
- Target: `B000EQU0NW` — Citizen Eco-Drive Corso Quartz Mens Watch, Stainless Steel with Leather strap, Classic, Brown (Model: BM8242-08E)
- Category: Clothing, Shoes & Jewelry > Men > Watches > Wrist Watches
- Longitudinal signal: P1/P2 reinforcement
- Directive: reinforce: I generally prefer dark colours and classic understated styling.
- Verified properties: style_color=classic brown [title: Classic, Brown]
- Why appropriate: The watch title directly specifies classic styling and brown colour.

#### S8: reinforce_compatible_budget

- Source: `public_0156` (buying, easy)
- Target: `B0C3KZXV4B` — adidas Alliance II Sackpack, Shadow Navy/Snowglobe/Dash Grey, One Size
- Category: Clothing, Shoes & Jewelry > Luggage & Travel Gear > Gym Bags > Drawstring Bags
- Longitudinal signal: P1/P2 reinforcement with target under approximately $120
- Directive: reinforce: I generally prefer dark neutral colours and understated styling.
- Verified properties: color=shadow navy and dash grey [title: Shadow Navy/Snowglobe/Dash Grey]; price=20.0 [price: 20.0]
- Why appropriate: Navy/grey colours and the numeric $20 price jointly support the scheduled history.

#### S9: strong_recent_reinforcement

- Source: `public_0160` (buying, easy)
- Target: `B01AAANF2Y` — Amazon Basics 4 Piece Packing Travel Organizer Cubes Set - Slim, Black
- Category: Clothing, Shoes & Jewelry > Luggage & Travel Gear > Travel Accessories > Packing Organizers
- Longitudinal signal: strong recent P1/P2 reinforcement
- Directive: reinforce: I strongly prefer dark colours and simple understated styling.
- Verified properties: color=black [details.Color: Black]; style=slim [details.Size: Slim]
- Why appropriate: The audited black slim organizer provides a strong, recent, controlled reinforcement.

#### S10: query_override_probe

- Source: `public_0145` (buying, easy)
- Target: `B00IJZZWGA` — BRIGHT STAR Low Cut Ankle Socks For Women - 30 Pairs of Athletic Socks For Running, Workout, Sports
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Active > Athletic Socks
- Longitudinal signal: explicit current override; portable price remains under approximately $120
- Directive: current_override: I normally choose dark understated items, but today I specifically want something bright, colourful, and sporty while still staying under approximately $120.
- Verified properties: use_case=athletic running socks [title: Athletic Socks For Running]; color_style=colourful designs [features[7]: colorful designs]; price=23.99 [price: 23.99]
- Why appropriate: A normal buying row explicitly supports colourful athletic socks and has a numeric price below $120.

### u3_distractor — distractor history

#### S1: useful_signal_p1

- Source: `public_0136` (buying, easy)
- Target: `B091F54MWM` — CAMPSNAIL 4 Pack Biker Shorts for Women High Waist - 5" Soft Summer Womens Shorts Spandex Workout Shorts for Running Athletic
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Active > Active Shorts
- Longitudinal signal: P1 first disclosure
- Directive: disclose: Across purchases, I generally prefer breathable materials.
- Verified properties: comfort=air permeability [features[7]: air permeability]
- Why appropriate: The active shorts explicitly promise air permeability.

#### S2: office_distractor

- Source: `public_0083` (buying, easy)
- Target: `B0BPMCJ1RD` — CHICZONE Plaid Shacket Jacket Womens Long Flannel Jacket Casual Lapel Button Down Tartan Trench Coats
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses & Button-Down Shirts
- Longitudinal signal: strong session-only office context
- Directive: session_only: For today's office event, I need a button-down collared layer.
- Verified properties: style=button-down collared [features[4]: turndown collar]
- Why appropriate: Button-down, collar, and office use are explicit catalogue properties.

#### S3: gym_distractor

- Source: `public_0095` (buying, easy)
- Target: `B09N78FT2W` — Free Leaper High Waisted Yoga Pants with Pockets for Women-Comfortable Running Seamless Leggings
- Category: Clothing, Shoes & Jewelry > Sport Specific Clothing > Yoga > Women > Leggings
- Longitudinal signal: strong session-only gym context
- Directive: session_only: For today's gym session, I need compression leggings with secure pockets.
- Verified properties: feature=compression [features[4]: fit compression]
- Why appropriate: The target explicitly names compression and workout use.

#### S4: useful_signal_p2

- Source: `public_0093` (buying, easy)
- Target: `B07PYB8F1G` — Hanes Women's Signature Breathe Cotton Brief Underwear 6-Pack
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lingerie > Panties > Briefs
- Longitudinal signal: P2 first disclosure under approximately $120
- Directive: disclose: Across purchases, I usually try to stay under approximately $120.
- Verified properties: price=10.47 [price: 10.47]
- Why appropriate: The target has a reliable low numeric price.

#### S5: rain_distractor

- Source: `public_0058` (buying, easy)
- Target: `B08L83YQTZ` — JTANIB Women Packable Rain Jacket Waterproof Lightweight Raincoat Hooded for Hiking Outdoor Travel
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Trench, Rain & Anoraks > Raincoats
- Longitudinal signal: strong session-only rain context
- Directive: session_only: For today's rainy hike, I need a waterproof hooded jacket.
- Verified properties: feature=waterproof [features[5]: waterproof rain jacket]; feature=hooded [features[6]: hooded hat design]
- Why appropriate: Waterproofing and a hood are directly verified.

#### S6: reinforce_useful_p1

- Source: `public_0114` (buying, easy)
- Target: `B07H34Z5V6` — Athlefit Women's Wedge Sneakers Hidden Heel Platform Wedge Booties Hidden Wedgie Sneakers
- Category: Clothing, Shoes & Jewelry > Women > Shoes > Fashion Sneakers
- Longitudinal signal: P1 reinforcement
- Directive: reinforce: I generally prefer breathable materials.
- Verified properties: comfort=breathable upper [features[4]: breathable soft upper]
- Why appropriate: A shoe context explicitly verifies a breathable upper.

#### S7: formal_distractor

- Source: `public_0066` (buying, easy)
- Target: `B0BFLFSB2Y` — GRAPENT Women's Plus Size Sequin 3/4 Sleeves Evening Gown Party Long Maxi Dress
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Dresses > Formal
- Longitudinal signal: strong session-only formal context
- Directive: session_only: For one formal dinner, I need a dressy evening gown.
- Verified properties: category=formal dress [categories: Formal]
- Why appropriate: The official category is Formal and features describe an evening party gown.

#### S8: travel_storage_distractor

- Source: `public_0101` (buying, easy)
- Target: `B07QMS8TX8` — Medical Cargo Pants for Men Workwear Originals, Zipper Fly Scrubs for Men 4000
- Category: Clothing, Shoes & Jewelry > Men > Uniforms, Work & Safety > Clothing > Medical > Scrub Bottoms
- Longitudinal signal: strong session-only storage context
- Directive: session_only: For one trip, I need cargo pants with lots of storage pockets.
- Verified properties: feature=seven storage pockets [features[7]: SEVEN POCKETS FOR LOTS OF STORAGE]
- Why appropriate: The cargo target explicitly verifies seven pockets and storage use.

#### S9: strong_recent_distractor

- Source: `public_0005` (buying, easy)
- Target: `B074G1JP8Z` — GLOBALWIN Women's Waterproof Winter Boots Snow Boots For Women
- Category: Clothing, Shoes & Jewelry > Boot Shop > Women > Outdoor & Work > Snow & Cold Weather
- Longitudinal signal: strong recent session-only distractor
- Directive: session_only: For one snowy trip, I strongly need insulated waterproof winter boots.
- Verified properties: feature=winter insulation [features[2]: Thermolite Insulation]; feature=waterproof [features[4]: Waterproof Seam-Sealed Construction]
- Why appropriate: Insulation and waterproof construction are unambiguous and irrelevant to the later T-shirt probe.

#### S10: distractor_probe

- Source: `public_0194` (buying, easy)
- Target: `B09WR1NZ48` — Graphic Tees for Women Short Sleeve Tshirts,Womens Summer Tops Crewneck Shirt Blouse
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
- Longitudinal signal: same S10 for H0/H1/H3/H5/H9 under approximately $120
- Directive: none
- Verified properties: comfort=breathable [features[3]: breathable]; price=19.99 [price: 19.99]
- Why appropriate: An ordinary breathable, inexpensive T-shirt makes office, gym, rain, formal, cargo, and snow requirements unnecessary.

### u4_negative — negative preference

#### S1: establish_n1

- Source: `public_0032` (buying, easy)
- Target: `B0834HZQZF` — IZZY + TOBY 100% Cotton Nightgowns for Women Soft Ladies Gowns Sleepwear Long Sleeveless Nightgown
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sleep & Lounge > Nightgowns & Sleepshirts
- Longitudinal signal: N1 explicit negative
- Directive: disclose: Across purchases, I do not want wool.
- Verified properties: material=100% cotton [features[0]: 100% Cotton]
- Why appropriate: The 100% cotton target offers a text-verifiable non-wool alternative.

#### S2: reinforce_n1

- Source: `public_0088` (buying, easy)
- Target: `B07Z6J5N6Y` — Amazon Essentials Women's Cotton Bikini Brief Underwear (Available in Plus Size), Multipacks
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lingerie > Panties > Bikinis
- Longitudinal signal: N1 reinforcement
- Directive: reinforce: I do not want wool in products I buy.
- Verified properties: material=95% cotton and 5% spandex [features[0]: 95% Cotton, 5% Spandex]
- Why appropriate: The exact composition verifies a cotton/spandex alternative without wool.

#### S3: neutral_task

- Source: `public_0132` (buying, easy)
- Target: `B08X2X83DW` — isotoner Women's Terry Slip on Clog Slipper with Memory Foam for Indoor/Outdoor Comfort
- Category: Clothing, Shoes & Jewelry > Women > Shoes > Slippers
- Longitudinal signal: none
- Directive: none
- Verified properties: category=slippers [categories: Slippers]
- Why appropriate: A normal slipper task separates negative introductions.

#### S4: establish_n2

- Source: `public_0199` (buying, easy)
- Target: `B089M57PSQ` — Boboking 100% Cotton Little Boys Briefs Soft Dinosaur Truck Toddler Underwear
- Category: Clothing, Shoes & Jewelry > Boys > Clothing > Underwear > Briefs
- Longitudinal signal: N2 explicit negative
- Directive: disclose: Across purchases, I avoid polyester-heavy clothing.
- Verified properties: material=100% cotton [features[2]: 100% Cotton]
- Why appropriate: The 100% cotton composition provides strong positive evidence for a non-polyester-heavy target.

#### S5: reinforce_n2

- Source: `public_0185` (buying, easy)
- Target: `B0BCW4QKV5` — MIOTAN Boy Shorts Underwear for Women High Waisted Panties Cotton Boxer Briefs 4 Pack
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lingerie > Panties > Boy Shorts
- Longitudinal signal: N2 reinforcement
- Directive: reinforce: I avoid polyester-heavy clothing.
- Verified properties: material=92% cotton and 8% spandex [features[0]: 92%Cotton/8%Spandex]
- Why appropriate: The exact 92% cotton composition is not polyester-heavy.

#### S6: neutral_task

- Source: `public_0118` (buying, easy)
- Target: `B09M72C8PG` — Angerella Women Vintage Polka Dot High Waisted Bathing Suits Bikini Set
- Category: Clothing, Shoes & Jewelry > Women > Clothing > Swimsuits & Cover Ups > Bikinis > Sets
- Longitudinal signal: none
- Directive: none
- Verified properties: material=82% nylon and 18% spandex [features[3]: 82%Nylon+18%Spandex]
- Why appropriate: A nylon/spandex swimsuit is a neutral task with explicit material evidence.

#### S7: establish_n3

- Source: `public_0028` (buying, easy)
- Target: `B0B9ZYDDZ1` — Buffway Slim Minimalist Front Pocket RFID Blocking Leather Wallets for Men Women
- Category: Clothing, Shoes & Jewelry > Men > Accessories > Wallets, Card Cases & Money Organizers > Wallets
- Longitudinal signal: N3 explicit negative
- Directive: disclose: Across purchases, I avoid very bright or neon colours.
- Verified properties: color=lichee black [details.Color: Lichee Black]
- Why appropriate: Target-level Lichee Black colour provides a clear non-neon alternative.

#### S8: reinforce_n1

- Source: `public_0188` (buying, easy)
- Target: `B0B5ZS2J2W` — CLUCI Crossbody Purses for Women, Medium Size Zipper Pocket Adjustable Strap, Soft Leather Women's Shoulder Handbags
- Category: Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Crossbody Bags
- Longitudinal signal: N1 reinforcement
- Directive: reinforce: I do not want wool.
- Verified properties: material=PU leather [features[0]: Pu Leather]
- Why appropriate: The bag's artificial/PU leather material is directly stated and non-wool.

#### S9: mature_negative_history

- Source: `public_0152` (buying, easy)
- Target: `B000EQU0NW` — Citizen Eco-Drive Corso Quartz Mens Watch, Stainless Steel with Leather strap, Classic, Brown (Model: BM8242-08E)
- Category: Clothing, Shoes & Jewelry > Men > Watches > Wrist Watches
- Longitudinal signal: mature N2/N3 reinforcement
- Directive: reinforce: I avoid polyester-heavy products.; I avoid very bright or neon colours.
- Verified properties: material_color=brown leather strap [title: Leather strap, Classic, Brown]
- Why appropriate: The classic brown leather target supplies direct non-polyester and non-neon evidence.

#### S10: negative_memory_probe

- Source: `public_0178` (buying, easy)
- Target: `B01FWQ8NH8` — Red Hot Chili Peppers Distressed Men's T-Shirt Black
- Category: Clothing, Shoes & Jewelry > Novelty & More > Clothing > Novelty > Men > Shirts > T-Shirts
- Longitudinal signal: same underspecified S10 for no-history/full-history negative-memory probe
- Directive: none
- Verified properties: material=100% cotton [features[0]: 100% Cotton]; color=black [title: T-Shirt Black]; price=17.99 [price: 17.99]
- Why appropriate: A 100% cotton black T-shirt violates none of N1/N2/N3 while same-category catalogue alternatives can contain wool, polyester, or bright colours.

## Rejected discovery candidates

- `public_0065` / `B0BSQ9TCYC` — Arctix Women's Essential Insulated Bib Overalls: No reliable numeric price, so it cannot support the portable budget tendency.
- `public_0106` / `B0776SVXW9` — Mens Socks Dress Cotton Socks Fashion Patterned Argyle Socks &Formal Business Socks Classic Cotton Dress Casual Socks for Men: The catalogue row mixes bright/colorful and formal/cotton claims across a product family; target-variant colour is ambiguous.
- `public_0053` / `B07TZK3GZK` — Passport Holder Cover Travel RFID Blocking Passport Cover Rose Gold Cute Flowers Passport Wallet with Elastic Band for Women: Search text mentions black while the exact title says rose gold flowers, making colour evidence contradictory.
- `public_0119` / `B0BBLR3QB2` — MIFORINES Ladies Summer Jelly Pillow-shaped Top Handle Handbag Candy Color Transparent Crystal Purse: Candy-color wording is broad and exact variant colour is not target-level verified.
- `public_0061` / `B08HCP9YTV` — 1pc Surgical Steel Piercing Ring for Nose Septum Cartilage Helix Tragus Conch Rook Daith Lobe 20g-18g-16g-14g-12g-10g 5mm-6mm-7mm-8mm-9mm-10mm-11mm-12mm-14mm-16mm Silver/Gold/Rose Gold/Black/Rainbow: The title lists many colour variants including black and rainbow; exact target colour cannot be isolated.
- `public_0094` / `B01L99SW78` — Ariat Fatbaby Western Boot – Women’s Leather Western Boots: No reliable numeric price and insufficient target-level material detail for a budget/material role.

## Discovery coverage

- Public buying rows inspected programmatically: 80
- Distinct selected public rows: 32
- Selected session assignments: 40
- Arbitrary catalogue-only targets: 0
- Attribute policy: category/department are structural; material, colour, style, use case, and price are used only when the session audit points to target-level catalogue evidence.

