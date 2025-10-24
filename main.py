from fastapi import FastAPI
from app.routers.inference_router import router

app = FastAPI()


app.include_router(router)