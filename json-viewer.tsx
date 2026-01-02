import React, { useState } from 'react';
import { ChevronRight, ChevronDown, Copy, Check } from 'lucide-react';
import { Button } from './button';

interface JsonViewerProps {
  data: any;
  defaultExpanded?: boolean;
  maxHeight?: string;
}

export const JsonViewer: React.FC<JsonViewerProps> = ({ 
  data, 
  defaultExpanded = true,
  maxHeight = '600px' 
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={handleCopy}
        className="absolute top-2 right-2 z-10 bg-gray-800 text-gray-200 border-gray-600 hover:bg-gray-700"
      >
        {copied ? (
          <>
            <Check className="h-3 w-3 mr-1" />
            Copied
          </>
        ) : (
          <>
            <Copy className="h-3 w-3 mr-1" />
            Copy
          </>
        )}
      </Button>
      <div 
        className="bg-gray-900 rounded-lg p-6 overflow-auto font-mono text-sm border border-gray-700"
        style={{ maxHeight, overflowY: 'auto', overflowX: 'auto' }}
      >
        <JsonNode data={data} name="root" defaultExpanded={defaultExpanded} level={0} />
      </div>
    </div>
  );
};

interface JsonNodeProps {
  data: any;
  name: string;
  defaultExpanded: boolean;
  level: number;
}

const JsonNode: React.FC<JsonNodeProps> = ({ data, name, defaultExpanded, level }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const indent = level * 20;

  if (data === null) {
    return (
      <div style={{ marginLeft: `${indent}px` }} className="py-0.5">
        <span style={{ color: '#c678dd' }}>{name}</span>
        <span style={{ color: '#abb2bf' }}>: </span>
        <span style={{ color: '#d19a66' }}>null</span>
      </div>
    );
  }

  if (data === undefined) {
    return (
      <div style={{ marginLeft: `${indent}px` }} className="py-0.5">
        <span style={{ color: '#c678dd' }}>{name}</span>
        <span style={{ color: '#abb2bf' }}>: </span>
        <span style={{ color: '#d19a66' }}>undefined</span>
      </div>
    );
  }

  if (typeof data === 'boolean') {
    return (
      <div style={{ marginLeft: `${indent}px` }} className="py-0.5">
        <span style={{ color: '#c678dd' }}>{name}</span>
        <span style={{ color: '#abb2bf' }}>: </span>
        <span style={{ color: '#d19a66' }}>{data.toString()}</span>
      </div>
    );
  }

  if (typeof data === 'number') {
    return (
      <div style={{ marginLeft: `${indent}px` }} className="py-0.5">
        <span style={{ color: '#c678dd' }}>{name}</span>
        <span style={{ color: '#abb2bf' }}>: </span>
        <span style={{ color: '#98c379' }}>{data}</span>
      </div>
    );
  }

  if (typeof data === 'string') {
    return (
      <div style={{ marginLeft: `${indent}px` }} className="py-0.5">
        <span style={{ color: '#c678dd' }}>{name}</span>
        <span style={{ color: '#abb2bf' }}>: </span>
        <span style={{ color: '#e5c07b' }}>"{data}"</span>
      </div>
    );
  }

  if (Array.isArray(data)) {
    const isEmpty = data.length === 0;
    
    return (
      <div style={{ marginLeft: `${indent}px` }}>
        <div className="flex items-center py-0.5 cursor-pointer hover:bg-gray-800 -mx-1 px-1 rounded" onClick={() => setIsExpanded(!isExpanded)}>
          {!isEmpty && (
            isExpanded ? 
              <ChevronDown className="h-3 w-3 mr-1" style={{ color: '#abb2bf' }} /> : 
              <ChevronRight className="h-3 w-3 mr-1" style={{ color: '#abb2bf' }} />
          )}
          {isEmpty && <span className="w-4 mr-1" />}
          <span style={{ color: '#c678dd' }}>{name}</span>
          <span style={{ color: '#abb2bf' }}>: </span>
          <span style={{ color: '#abb2bf' }}>[</span>
          {!isExpanded && !isEmpty && (
            <span style={{ color: '#5c6370' }} className="text-xs ml-1">{data.length} items</span>
          )}
          {isEmpty && <span style={{ color: '#abb2bf' }}>]</span>}
        </div>
        {isExpanded && !isEmpty && (
          <>
            {data.map((item, index) => (
              <JsonNode
                key={index}
                data={item}
                name={`[${index}]`}
                defaultExpanded={false}
                level={level + 1}
              />
            ))}
            <div style={{ marginLeft: `${indent}px`, color: '#abb2bf' }} className="py-0.5">]</div>
          </>
        )}
      </div>
    );
  }

  if (typeof data === 'object') {
    const keys = Object.keys(data);
    const isEmpty = keys.length === 0;

    return (
      <div style={{ marginLeft: `${indent}px` }}>
        <div className="flex items-center py-0.5 cursor-pointer hover:bg-gray-800 -mx-1 px-1 rounded" onClick={() => setIsExpanded(!isExpanded)}>
          {!isEmpty && (
            isExpanded ? 
              <ChevronDown className="h-3 w-3 mr-1" style={{ color: '#abb2bf' }} /> : 
              <ChevronRight className="h-3 w-3 mr-1" style={{ color: '#abb2bf' }} />
          )}
          {isEmpty && <span className="w-4 mr-1" />}
          <span style={{ color: '#c678dd' }}>{name}</span>
          <span style={{ color: '#abb2bf' }}>: </span>
          <span style={{ color: '#abb2bf' }}>{'{'}</span>
          {!isExpanded && !isEmpty && (
            <span style={{ color: '#5c6370' }} className="text-xs ml-1">{keys.length} keys</span>
          )}
          {isEmpty && <span style={{ color: '#abb2bf' }}>{'}'}</span>}
        </div>
        {isExpanded && !isEmpty && (
          <>
            {keys.map((key) => (
              <JsonNode
                key={key}
                data={data[key]}
                name={key}
                defaultExpanded={level < 1}
                level={level + 1}
              />
            ))}
            <div style={{ marginLeft: `${indent}px`, color: '#abb2bf' }} className="py-0.5">{'}'}</div>
          </>
        )}
      </div>
    );
  }

  return null;
};
