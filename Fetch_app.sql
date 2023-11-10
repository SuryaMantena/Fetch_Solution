SELECT * FROM brand_category;
SELECT
    t1.*,
    t2.BRAND_BELONGS_TO_CATEGORY,
    t3.IS_CHILD_CATEGORY_TO
FROM
    offfer_retailer_nonascii t1
LEFT JOIN
    brand_category t2
ON
    t1.Brand = t2.Brand
LEFT JOIN
    categories t3
ON
    t2.BRAND_BELONGS_TO_CATEGORY = t3.PRODUCT_CATEGORY;
