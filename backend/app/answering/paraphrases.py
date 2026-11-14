"""Hand-written question paraphrases with expected templates (EN slang, FI, SV, traps, out of scope).

Used by scripts/routing_experiment.py and the benchmark (app.bench). "none" = unsupported: the answer must be
not_found.
"""

# (question, accepted labels). "none" = must not call the tool.
PARAPHRASES: list[tuple[str, set[str]]] = [
    # casual English
    ("how many windows does this place have?", {"windows"}),
    ("Number of doors in the apartment?", {"doors"}),
    ("Is this a two-storey house? How many levels are drawn?", {"floors"}),
    ("How many toilets are there? I mean the actual toilet bowls.", {"wc_seats"}),
    ("Count the wash basins and kitchen sinks for me.", {"sinks"}),
    ("How many places can you take a shower or a bath?", {"showers_baths"}),
    ("Does the flat have saunas? How many?", {"saunas"}),
    ("How many kitchenettes or kitchens are there?", {"kitchens"}),
    ("How many rooms are bedrooms?", {"bedrooms"}),
    ("Number of washrooms, excluding the separate toilets?", {"bathrooms"}),
    ("How many standalone WCs are there?", {"wc_rooms"}),
    ("How many balconies or terraces with glazing?", {"balconies"}),
    ("How many bedrooms does this unit have", {"bedrooms"}),
    ("total count of baths + showers", {"showers_baths"}),
    # Finnish, incl. slang / other words than the dataset
    ("Montako ikkunaa asunnossa on?", {"windows"}),
    ("Montako ovea tässä asunnossa on?", {"doors"}),
    ("Montako tasoa piirustuksessa on?", {"floors"}),
    ("Montako vessanpönttöä on piirretty?", {"wc_seats"}),
    ("Montako lavuaaria ja tiskiallasta on yhteensä?", {"sinks"}),
    ("Montako suihkua tai ammetta asunnossa on?", {"showers_baths"}),
    ("Onko asunnossa saunaa? Montako?", {"saunas"}),
    ("Montako keittokomeroa tai keittiötä on?", {"kitchens"}),
    ("Montako makkaria kämpässä on?", {"bedrooms"}),
    ("Montako kylppäriä on, erillisiä vessoja ei lasketa?", {"bathrooms"}),
    ("Montako erillistä vessaa asunnossa on?", {"wc_rooms"}),
    ("Montako parveketta tai lasitettua parveketta on?", {"balconies"}),
    # Swedish
    ("Hur många fönster finns på planritningen?", {"windows"}),
    ("Hur många dörrar finns det?", {"doors"}),
    ("Hur många våningar visar ritningen?", {"floors"}),
    ("Hur många toalettstolar är ritade?", {"wc_seats"}),
    ("Hur många handfat och diskhoar finns det totalt?", {"sinks"}),
    ("Hur många duschar och badkar finns det?", {"showers_baths"}),
    ("Hur många bastur finns det?", {"saunas"}),
    ("Hur många kök finns det?", {"kitchens"}),
    ("Hur många sovrum har lägenheten?", {"bedrooms"}),
    ("Hur många badrum finns det, separata WC ej medräknade?", {"bathrooms"}),
    ("Hur många separata WC finns det?", {"wc_rooms"}),
    ("Hur många balkonger finns det?", {"balconies"}),
    # traps: keywords of one template, but asking another / a composition / something unsupported
    ("What is the area of the kitchen?", {"none"}),
    ("How wide is the front door?", {"none"}),
    ("Are there windows in the bedroom?", {"none"}),
    ("How many sinks are there in the bathroom?", {"none"}),
    ("How many bathrooms have a shower?", {"none"}),
    ("Is the sauna next to the bathroom?", {"none"}),
    ("How many doors open onto the balcony?", {"none"}),
    ("Mikä on keittiön pinta-ala?", {"none"}),
    ("Hur stor är balkongen i kvadratmeter?", {"none"}),
    ("How many sinks are in the kitchen?", {"none"}),
    ("Kuinka monta ikkunaa makuuhuoneessa on?", {"none"}),
    # out of scope
    ("What is the total floor area?", {"none"}),
    ("What scale is this drawing?", {"none"}),
    ("How many radiators are shown?", {"none"}),
    ("How many rooms are there in total?", {"none"}),
    ("Kuinka monta neliötä asunnossa on?", {"none"}),
    ("Who drew this plan?", {"none"}),
    ("How many parking spaces are there?", {"none"}),
    ("Hur många rum finns det?", {"none"}),
]
