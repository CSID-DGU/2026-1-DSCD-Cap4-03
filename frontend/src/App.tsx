import { BrowserRouter, Routes, Route } from "react-router-dom";

import Navbar from "./components/layout/Navbar";
import Footer from "./components/layout/Footer";
import ProtectedRoute from "./components/ProtectedRoute";
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
import PrivacyPage from "./pages/PrivacyPage";
import TermsPage from "./pages/TermsPage";
import VanityMainPage from "./pages/VanityMainPage";
import SkinMatchPage from "./pages/SkinMatchPage";
import VanityRoutineBudgetPage from "./pages/VanityRoutineBudgetPage";
import VanityRoutinePage from "./pages/VanityRoutinePage";
import VanityRoutineHistoryPage from "./pages/VanityRoutineHistoryPage";

import { AuthProvider } from "./context/AuthProvider";
import ScrollToTop from "./components/ScrollToTop";

const P = ({ children }: { children: React.ReactNode }) => (
  <ProtectedRoute>{children}</ProtectedRoute>
);

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <ScrollToTop />
        <Navbar />
        <Routes>
          {/* 공개 */}
          <Route path="/" element={<MainPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/privacy" element={<PrivacyPage />} />
          <Route path="/terms" element={<TermsPage />} />

          {/* 로그인 필요 */}
          <Route path="/userinfo"          element={<P><UserInfoPage /></P>} />
          <Route path="/diagnosis"         element={<P><DiagnosisPage /></P>} />
          <Route path="/loading"           element={<P><LoadingPage /></P>} />
          <Route path="/analysis-history"  element={<P><AnalysisHistoryPage /></P>} />
          <Route path="/analysis"          element={<P><AnalysisResultPage /></P>} />
          <Route path="/routine/budget"    element={<P><BudgetPage /></P>} />
          <Route path="/routine/result"    element={<P><RoutinePage /></P>} />
          <Route path="/routine-history"   element={<P><RoutineHistoryPage /></P>} />
          <Route path="/products"          element={<P><ProductListPage /></P>} />
          <Route path="/products/:id"      element={<P><ProductDetailPage /></P>} />
          <Route path="/mypage"            element={<P><MyPage /></P>} />
          {/* 스킨핏 — 내 화장대 */}
          <Route path="/vanity"                    element={<P><VanityMainPage /></P>} />
          <Route path="/vanity/skin-match"         element={<P><SkinMatchPage /></P>} />
          <Route path="/vanity/routine/budget"     element={<P><VanityRoutineBudgetPage /></P>} />
          <Route path="/vanity/routine"            element={<P><VanityRoutinePage /></P>} />
          <Route path="/vanity/routine/history"    element={<P><VanityRoutineHistoryPage /></P>} />
        </Routes>
        <Footer />
      </BrowserRouter>
    </AuthProvider>
  );
}
