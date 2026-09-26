export function MockBanner({ show }: { show: boolean }) {
  if (!show) return null;
  return <div className="mock-banner">MOCK DATA — values are synthetic, not experimental results</div>;
}
