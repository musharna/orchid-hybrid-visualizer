"""Canonical list of 27 named Cattleya hybrid crosses.

All parentage verified against the RHS International Orchid Register (2026-03).
This is the single source of truth — imported by regen, audit, and UI scripts.
"""

# Each entry: (filename_stem, display_name, ancestry_dict, reference_url)
# reference_url is for the primary single-reference audit image (may be None)
CROSSES = [
    ("C_Hardyana", "C. Hardyana", {"dowiana": 50, "warscewiczii": 50},
     "https://cdn11.bigcommerce.com/s-ookf1bkiza/images/stencil/608w/products/22705/26380/chardy4b__99079.1752772362.jpg"),
    ("C_Brabantiae", "C. Brabantiae", {"aclandiae": 50, "loddigesii": 50},
     "https://www.orchids.org/rails/active_storage/representations/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBK2hBQVE9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--f1a1ee67aeb9046c170cf7e4b50eacc2a5b9e4fe/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCam9MY21WemFYcGxTU0lNTVRBd2VERXdNQVk2QmtWVSIsImV4cCI6bnVsbCwicHVyIjoidmFyaWF0aW9uIn19--a9526fc72018ec9f872badfc17d9b9f850bf1b7b/Cattleya%20Brabantiae%20'El%20Toro'%20AM%20AOS%202025f%2001.jpg"),
    ("C_Interglossa", "C. Interglossa", {"amethystoglossa": 50, "intermedia": 50},
     "https://www.orchidroots.com/static/utils/images/hybrid/hyb_000348984_100072348.jpg"),
    ("C_Empress_Frederick", "C. Empress Frederick", {"dowiana_aurea": 50, "mossiae": 50},
     "https://www.orchidroots.com/static/utils/images/hybrid/hyb_000433633_100037810.jpg"),
    ("C_guatemalensis", "C. guatemalensis", {"aurantiaca": 50, "skinneri": 50},
     "https://s3.amazonaws.com/iof-grexes3/177619.jpg"),
    ("C_Canhamiana", "C. Canhamiana", {"purpurata": 50, "mossiae": 50},
     "https://www.laforestaorchids.com/cdn/shop/products/cattleya-mossiae-x-cattleya-purpurata-cattleya-la-foresta-orchids-681149.jpg?v=1725538778"),
    ("C_Iris", "C. Iris", {"bicolor": 50, "dowiana": 50},
     "https://cdn4.volusion.store/hyokq-tvfom/v/vspfiles/photos/MC1Iris-2.jpg?v-cache=1765460947"),
    ("C_Pittiae", "C. Pittiae", {"harrisoniana": 50, "schilleriana": 50},
     "https://www.orchidroots.com/static/utils/images/hybrid/hyb_000070045_000002524.jpg"),
    ("C_Leda", "C. Leda", {"dowiana": 50, "percivaliana": 50},
     "https://s3.amazonaws.com/iof-grexes3/108546.jpg"),
    ("C_Landate", "C. Landate", {"aclandiae": 50, "guttata": 50},
     "https://www.orchidroots.com/static/utils/images/hybrid/mcxaiDqYfwaxrmQ3QoRR8v_IMG_7485.jpeg"),
    ("C_Wendlandiana", "C. Wendlandiana", {"bowringiana": 50, "warscewiczii": 50}, None),
    ("C_Dupreana", "C. Dupreana", {"warneri": 50, "warscewiczii": 50},
     "https://cdn11.bigcommerce.com/s-ookf1bkiza/images/stencil/608w/products/22753/26457/cattdupreanacoeruleam__19926.1752772370.jpg"),
    ("C_Dolosa", "C. Dolosa", {"loddigesii": 50, "walkeriana": 50},
     "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiBoe7mVE5EselZYuDkV6OJ5QJTemZK2_S7FIruLf6ZUdyB_eTOzQ8Stg0o7dL_rSujd63eMwYPU7YOs4iEOA9gzOELSmMTZyMLseyUN9Gi-kt_L-o0ZPxvr9QR0mvlMYDxo3drEkcLtAI/s1600/Orqu%25C3%25ADdea-Cattleya-Dolosa.jpg"),
    ("C_Claesiana", "C. Claesiana", {"intermedia": 50, "loddigesii": 50},
     "https://s3.amazonaws.com/iof-grexes3/146683.jpg"),
    ("C_Browniae", "C. Browniae", {"bowringiana": 50, "harrisoniana": 50}, None),
    ("C_Quinquecolor", "C. Quinquecolor", {"aclandiae": 50, "forbesii": 50},
     "https://www.sborchid.com/orchidphotos/Cattleya/c_quinquecolor__4w.jpg"),
    ("C_Chocolate_Drop", "C. Chocolate Drop", {"guttata": 50, "aurantiaca": 50},
     "https://www.orchidroots.com/static/utils/images/hybrid/BKmPgMRMXgQx2E8AywZrHQ_Cattlianthe_Chocolate_Drop.jpg"),
    ("C_Triumphans", "C. Triumphans", {"dowiana": 50, "rex": 50},
     "https://cdn.imagearchive.com/slippertalk/data/attachments/32/32920-935f1323004977fa31f5728699a1dbba.jpg"),
    ("C_Venus", "C. Venus", {"dowiana": 75, "bicolor": 25}, None),  # complex: dowiana × Iris(1901)
    ("C_Whitei", "C. Whitei", {"schilleriana": 50, "warneri": 50},
     "https://s3.amazonaws.com/iof-grexes3/131022.jpg"),
    ("C_Measuresiana", "C. Measuresiana", {"aclandiae": 50, "walkeriana": 50},
     "https://www.krullsmith.com/images/product/large/385_1_.jpg"),
    ("C_Elegans", "C. Elegans", {"tigrina": 50, "purpurata": 50},
     "https://s3.amazonaws.com/iof-grexes3/133806.jpg"),
    ("C_Peetersii", "C. Peetersii", {"lawrenceana": 50, "warscewiczii": 25, "purpurata": 25},
     "https://s3.amazonaws.com/iof-grexes3/56219.jpg"),  # complex: Callistoglossa × lawrenceana
    ("C_Minucia", "C. Minucia", {"loddigesii": 50, "warscewiczii": 50}, None),
    ("C_Brymeriana", "C. Brymeriana", {"violacea": 50, "wallisii": 50},
     "https://s3.amazonaws.com/iof-grexes3/52093.jpg"),  # natural hybrid
    ("C_Portia", "C. Portia", {"bowringiana": 50, "labiata": 50},
     "https://cdn11.bigcommerce.com/s-ookf1bkiza/images/stencil/608w/products/23242/27313/cportap4b__53407.1752772443.jpg"),
    ("C_Mem_Albert_Heinecke", "C. Mem. Albert Heinecke",
     {"dowiana": 69, "bicolor": 16, "tenebrosa": 13, "warscewiczii": 3},
     "https://s3.amazonaws.com/iof-grexes3/56357.jpg"),  # complex hybrid
]

# Convenience accessors
CROSSES_SIMPLE = [(name, ancestry) for name, _, ancestry, _ in CROSSES]
CROSSES_WITH_DISPLAY = [(name, display, ancestry) for name, display, ancestry, _ in CROSSES]
CROSS_REF_URLS = {name: url for name, _, _, url in CROSSES if url}

# Map from cross filename stem to multi-ref directory name
CROSS_TO_REF_DIR = {
    "C_Hardyana": "Hardyana",
    "C_Brabantiae": "Brabantiae",
    "C_Interglossa": "Interglossa",
    "C_Empress_Frederick": "Empress_Frederick",
    "C_guatemalensis": "guatemalensis",
    "C_Canhamiana": "Canhamiana",
    "C_Iris": "Iris",
    "C_Pittiae": "Pittiae",
    "C_Landate": "Landate",
    "C_Dupreana": "Dupreana",
    "C_Dolosa": "Dolosa",
    "C_Claesiana": "Claesiana",
    "C_Quinquecolor": "Quinquecolor",
    "C_Chocolate_Drop": "Chocolate_Drop",
    "C_Triumphans": "Triumphans",
    "C_Portia": "Portia",
    "C_Elegans": "Elegans",
    "C_Measuresiana": "Measuresiana",
    "C_Whitei": "Whitei",
    "C_Brymeriana": "Brymeriana",
    "C_Mem_Albert_Heinecke": "Mem_Albert_Heinecke",
    "C_Leda": "Leda",
    "C_Peetersii": "Peetersii",
    "C_Browniae": "Browniae",
    "C_Minucia": "Minucia",
}
