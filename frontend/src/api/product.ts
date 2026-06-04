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

const productListCache = new Map<string, ProductSummary[]>();
const productListPending = new Map<string, Promise<ProductSummary[]>>();
const PRODUCT_LIST_STORAGE_PREFIX = 'rouple_product_list:';

function productListKey(category?: string) {
  return category && category !== '전체' ? category : '__all__';
}

function productListPath(category?: string) {
  return category && category !== '전체'
    ? `/products?category=${encodeURIComponent(category)}`
    : '/products';
}

function getCachedProductList(category?: string) {
  const key = productListKey(category);
  const storageKey = `${PRODUCT_LIST_STORAGE_PREFIX}${key}`;

  const cached = productListCache.get(key);
  if (cached) return Promise.resolve(cached);

  try {
    const raw = sessionStorage.getItem(storageKey);
    if (raw) {
      const parsed = JSON.parse(raw) as ProductSummary[];
      productListCache.set(key, parsed);
      return Promise.resolve(parsed);
    }
  } catch {
    // Storage can fail on some mobile/private browser modes.
  }

  const pending = productListPending.get(key);
  if (pending) return pending;

  const request = api.get<ProductSummary[]>(productListPath(category))
    .then((products) => {
      productListCache.set(key, products);
      try {
        sessionStorage.setItem(storageKey, JSON.stringify(products));
      } catch {
        // Large product lists may exceed storage quota; memory cache still helps.
      }
      return products;
    })
    .finally(() => {
      productListPending.delete(key);
    });

  productListPending.set(key, request);
  return request;
}

export function clearProductListCache() {
  productListCache.clear();
  productListPending.clear();
  try {
    Object.keys(sessionStorage)
      .filter((key) => key.startsWith(PRODUCT_LIST_STORAGE_PREFIX))
      .forEach((key) => sessionStorage.removeItem(key));
  } catch {
    // no-op
  }
}

export const productApi = {
  getList: (category?: string) => getCachedProductList(category),

  getDetail: (productId: number) =>
    api.get<ProductDetail>(`/products/${productId}`),

  addWishlist: (productId: number) =>
    api.post<{ product_id: number; saved: boolean }>(`/wishlist/${productId}`),

  removeWishlist: (productId: number) =>
    api.delete<{ product_id: number; saved: boolean }>(`/wishlist/${productId}`),

  getWishlist: () =>
    api.get<{ items: ProductSummary[] }>('/users/me/wishlist'),
};
