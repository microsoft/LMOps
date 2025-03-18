from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RagPath:

    query: str
    past_subqueries: Optional[List[str]]
    past_subanswers: Optional[List[str]]
    past_doc_ids: Optional[List[List[str]]]
