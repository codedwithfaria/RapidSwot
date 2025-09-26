import { useState } from 'react';
import { AGUIProvider } from '../lib/agui';
import { ChatInterface } from '../components/ChatInterface';
import { Sidebar } from '../components/Sidebar';
import { Toolbar } from '../components/Toolbar';

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <AGUIProvider endpoint={process.env.NEXT_PUBLIC_AGUI_ENDPOINT || 'ws://localhost:8000/ws'}>
      <div className="flex h-screen">
        {/* Sidebar */}
        <Sidebar 
          isOpen={sidebarOpen} 
          onToggle={() => setSidebarOpen(!sidebarOpen)} 
        />

        {/* Main content */}
        <main className="flex-1 flex flex-col">
          {/* Toolbar */}
          <Toolbar />

          {/* Chat interface */}
          <div className="flex-1 overflow-hidden">
            <ChatInterface />
          </div>
        </main>
      </div>
    </AGUIProvider>
  );
}