"""Deterministic question answering over IFC (BIM) models.

model.py   IFC file -> IfcModel evidence (header, project / site / building / storeys, every element with its class,
           storey, type, materials, common properties and SI-normalised quantities; spaces with resolved areas).
tools.py   general tools over an IfcModel (summary, storeys, select / count / group elements, find spaces, sum a
           quantity, element details); every result carries the GlobalIds it came from.
answer.py  question -> query plan -> tools -> Answer (citations = PlanItems whose id is the IFC GlobalId).
eval.py    score on questions.jsonl.
"""
