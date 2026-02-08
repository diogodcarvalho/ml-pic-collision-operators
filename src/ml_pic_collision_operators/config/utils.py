from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # reject unknown fields
        validate_assignment=True,  # re-validate on attribute set
        frozen=True,
    )
