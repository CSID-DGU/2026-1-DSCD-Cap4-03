from fastapi import APIRouter

from app.api.routes import auth, files, health, images, products, recommendations, routines, skin_analysis, users, vanity, wishlist


api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(images.router, tags=["images"])
api_router.include_router(skin_analysis.router, tags=["skin-analysis"])
api_router.include_router(recommendations.router, tags=["recommendations"])
api_router.include_router(products.router, tags=["products"])
api_router.include_router(wishlist.router, tags=["wishlist"])
api_router.include_router(routines.router, tags=["routines"])
api_router.include_router(vanity.router, tags=["vanity"])
