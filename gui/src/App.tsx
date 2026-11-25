function App() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">RAG GUI</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Retriever & Reranker Experimentation Tool</p>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Welcome to RAG GUI</h2>
          <p className="text-gray-500 dark:text-gray-400 max-w-2xl mx-auto">
            This application provides tools for evaluating retrieval and reranking strategies in RAG pipelines.
            Configure your query, run retrieval, and compare results side-by-side.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
