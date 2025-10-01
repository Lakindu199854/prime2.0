# Save this in a file, perhaps named 'schema.py'
from pydantic import BaseModel, Field
from typing import List, Optional
# from dicttoxml import dicttoxml # Removed, as we use a separate converter now

# --- 1. Nested Models (Components of the final structure) ---

class AbstractData(BaseModel):
    English: Optional[str] = Field(None, description="The abstract text in English.")
    Chinese: Optional[str] = Field(None, description="The abstract text in Chinese (if available).")

class Contributor(BaseModel):
    Name: str = Field(..., description="The contributor's name (e.g., 'Zhang Xinfeng').If the name is in multiple languages get all")
    AffiliationRef: Optional[str] = Field(None, description="Reference ID linking to the Affiliation model (e.g., '1').")
    CorrespondingAuthor: Optional[bool] = Field(None, description="Set to true if this is the corresponding author.")
    # The XML needed 'ChineseName', but Pydantic uses clean names, we'll map this in XML conversion

class Affiliation(BaseModel):
    ID: str = Field(..., description="A unique identifier for the affiliation (e.g., '1', '2').")
    Description: str = Field(..., description="The full address and institutional name.")

class TitleInformation(BaseModel):
    ConferenceArticleTitle: str
    SourceTitle: Optional[str] = None
    IssueTitle: Optional[str] = None
    AbbreviatedSourceTitle: Optional[str] = None

class CitationInformation(BaseModel):
    DOI: Optional[str] = None
    PublicationYear: Optional[int] = None
    VolumeNumber: Optional[str] = None
    PageNumber: Optional[str] = None
    IssueText_IssueNumber: Optional[str] = Field(None, alias="IssueText / Issue Number")
    PublicationNumber: Optional[str] = None

class SourceInformation(BaseModel):
    Publisher: Optional[str] = None
    VolumeTitle: Optional[str] = None
    ISBN_ISSBN: Optional[str] = Field(None, alias="ISBN (ISSBN)")
    ISSN: Optional[str] = None
    Contributors: List[Contributor] = Field(default_factory=list)
    Affiliations: List[Affiliation] = Field(default_factory=list)

class ConferenceEventInformation(BaseModel):
    ConferenceName: Optional[str] = None
    ConferenceDate: Optional[str] = None
    ConferenceLocation: Optional[str] = None
    ConferenceSponsors: Optional[str] = None
    DocumentType: Optional[str] = None

# --- 2. Main Top-Level Model ---

class ArticleData(BaseModel):
    """The root structure for all extracted article data."""
    Abstract: AbstractData
    TitleInformation: TitleInformation
    CitationInformation: CitationInformation
    SourceInformation: SourceInformation
    ConferenceEventInformation: ConferenceEventInformation
