from fastapi import FastAPI

from routers import sound

app = FastAPI()

# Endpoints
app.include_router(sound.router)


@app.get("/")
async def root():
    return {"message": "C.U.S.T.O.M. open API: "
            "https://github.com/murielsan/CUSTOM/"
            }
