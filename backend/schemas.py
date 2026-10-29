from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Scene(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sky: Literal['dawn','day','night','sunset']
    landform: Literal['plains','mountains','volcano']
    ground: Literal['grass','sand','rock']
    water: Literal['none','river','lake','ocean']
    objects: Literal['none','trees','settlement']


class PromptRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=3,max_length=1000)


class ScenePlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scene: Scene
    unsupported: list[str] = Field(max_length=20)
    explanation: str = Field(max_length=1000)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scene: Scene
    seed: int | None = Field(default=None,ge=0,le=2147483647,strict=True)
    temperature: float = Field(default=1.,ge=.5,le=1.2,allow_inf_nan=False)


class GenerationResult(BaseModel):
    image_base64: str
    mime_type: Literal['image/png'] = 'image/png'
    width: int = 64
    height: int = 64
    seed: int
    model_version: str
    checkpoint_sha256: str
    seconds: float
    scene: Scene
