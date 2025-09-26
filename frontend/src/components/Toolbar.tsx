export function Toolbar() {
  return (
    <div className="h-14 border-b flex items-center px-4 justify-between">
      {/* Left section */}
      <div className="flex items-center space-x-4">
        <button className="p-2 hover:bg-gray-100 rounded">
          New Chat
        </button>
        <button className="p-2 hover:bg-gray-100 rounded">
          Clear Context
        </button>
      </div>

      {/* Right section */}
      <div className="flex items-center space-x-4">
        <button className="p-2 hover:bg-gray-100 rounded">
          Settings
        </button>
        <button className="p-2 hover:bg-gray-100 rounded">
          Help
        </button>
      </div>
    </div>
  );
}