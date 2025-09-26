import { useUIComponents } from '../lib/agui';

export function Sidebar({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
  const components = useUIComponents();

  return (
    <div
      className={`${
        isOpen ? 'w-64' : 'w-0'
      } transition-all duration-300 bg-gray-800 text-white overflow-hidden`}
    >
      <div className="p-4">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold">RapidSwot</h2>
          <button
            onClick={onToggle}
            className="p-2 hover:bg-gray-700 rounded"
          >
            {isOpen ? '←' : '→'}
          </button>
        </div>

        {/* Dynamic UI Components */}
        <div className="space-y-4">
          {components.map((component, index) => (
            <div key={index} className="p-2 bg-gray-700 rounded">
              {/* Render dynamic components based on type */}
              {component.type === 'button' && (
                <button
                  onClick={() => {
                    // Handle component action
                  }}
                  className="w-full text-left hover:bg-gray-600 p-2 rounded"
                >
                  {component.label}
                </button>
              )}
              {component.type === 'status' && (
                <div className="flex items-center space-x-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      component.value === 'active' ? 'bg-green-500' : 'bg-red-500'
                    }`}
                  />
                  <span>{component.label}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}