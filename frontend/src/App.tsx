import { BrowserRouter, Routes, Route } from "react-router-dom";

import Navbar from "./components/layout/Navbar";
import MainPage from "./pages/MainPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import UserInfoPage from "./pages/UserInfoPage";
import DiagnosisPage from "./pages/DiagnosisPage";
import LoadingPage from "./pages/LoadingPage";
import AnalysisHistoryPage from "./pages/AnalysisHistoryPage";
import AnalysisResultPage from "./pages/AnalysisResultPage";
import RoutineHistoryPage from "./pages/RoutineHistoryPage";
import BudgetPage from "./pages/BudgetPage";
import RoutinePage from "./pages/RoutinePage";
import ProductListPage from "./pages/ProductListPage";
import ProductDetailPage from "./pages/ProductDetailPage";
import MyPage from "./pages/MyPage";

import { AuthProvider } from "./context/AuthProvider";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Navbar />
        <Routes>
          {/* 기존 */}
          <Route path="/" element={<MainPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/userinfo" element={<UserInfoPage />} />
          <Route path="/diagnosis" element={<DiagnosisPage />} />

          {/* 로딩 */}
          <Route path="/loading" element={<LoadingPage />} />

          {/* 중간 페이지 */}
          <Route path="/analysis-history" element={<AnalysisHistoryPage />} />
          <Route path="/routine-history" element={<RoutineHistoryPage />} />

          {/* 결과 / 루틴 */}
          <Route path="/analysis" element={<AnalysisResultPage />} />
          <Route path="/routine/budget" element={<BudgetPage />} />
          <Route path="/routine/result" element={<RoutinePage />} />

          {/* 제품 */}
          <Route path="/products" element={<ProductListPage />} />
          <Route path="/products/:id" element={<ProductDetailPage />} />

          {/* 마이 */}
          <Route path="/mypage" element={<MyPage />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
