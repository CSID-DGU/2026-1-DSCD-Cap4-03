import { api } from './client';

export interface ProductSummary {
  product_id: number;
  brand_name: string;
  product_name: string;
  category: string;
  price: number;
  image_url: string;
  tags: string[];
}

export interface ProductDetail extends ProductSummary {
  ingredients: string[];
  pros: string[];
  cons: string[];
  how_to_use: string;
  apply_time: string;
}

export const productApi = {
  getList: (category?: string) =>
    api.get<ProductSummary[]>(
      category && category !== '전체' ? `/products?category=${encodeURIComponent(category)}` : '/products'
    ),

  getDetail: (productId: number) =>
    api.get<ProductDetail>(`/products/${productId}`),

  addWishlist: (productId: number) =>
    api.post<{ product_id: number; saved: boolean }>(`/wishlist/${productId}`),

  removeWishlist: (productId: number) =>
    api.delete<{ product_id: number; saved: boolean }>(`/wishlist/${productId}`),

  getWishlist: () =>
    api.get<{ items: ProductSummary[] }>('/users/me/wishlist'),
};
