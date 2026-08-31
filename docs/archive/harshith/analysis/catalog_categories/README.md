# Catalogue category analysis

This directory contains an exact extraction of the `categories` field from all
50,000 products in `data/catalog.jsonl`.

## Outputs

- `product_categories.csv`: one row per product, including ASIN, title, path,
  depth, and the original category list serialized as JSON.
- `category_label_occurrences.csv`: one row per distinct category label, counted
  wherever the label appears in a product path. The depth breakdown avoids
  conflating labels such as `Women`, which occur at several levels.
- `category_hierarchy_nodes.csv`: one row per distinct path prefix. This is the
  safe source for hierarchical filtering because the same label can occur under
  different parents.
- `category_tree.json`: nested form of the path-prefix hierarchy.
- `summary.json`: coverage and cardinality totals.

Regenerate all outputs with:

```bash
python3 techjam-conversational-search/analysis/catalog_categories/analyze_catalog_categories.py
```

## Main findings

- Every product has a non-empty category path.
- There are 863 distinct labels, 1,628 distinct full paths, and 1,832 distinct
  path-prefix nodes.
- Paths range from 2 to 8 levels deep.
- The root is almost entirely `Clothing, Shoes & Jewelry` (49,990 products),
  with 10 products under `Shoe, Jewelry & Watch Accessories`.
- Under the main root, the dominant audience branches are `Women` (26,406),
  `Men` (9,901), `Girls` (1,716), `Boys` (1,101), and `Baby` (1,031).
- The taxonomy also contains storefront, campaign, and test nodes such as
  `Westlake`, `Boot Shop`, `Toddler Test`, and `Swimwear TEST`. These should not
  automatically become user-facing facets.

## Filtering implication

Use the full path prefix for exact catalogue elimination, not a bare label. A
bare label can occur at multiple depths and under multiple parents. For a clean
user-facing filter layer, parse paths into independent facets:

1. Audience: Women, Men, Girls, Boys, Baby.
2. Product family: Clothing, Shoes, Jewelry, Watches, Accessories, Handbags,
   Luggage, Costumes, Uniforms/workwear.
3. Product subtype: the remaining deeper path, such as Shoes > Athletic >
   Running > Road Running.

This faceted representation is safer than forcing the raw catalogue into one
canonical tree: audience and product family appear in different orders across
some paths, while promotional/test branches cut across both.
