// File Tree Component - Displays hierarchical file structure with code preview
import React, { useState } from 'react';
import { ChevronRight, ChevronDown, File, Folder, FolderOpen, Code, FileText, Image, Archive } from 'lucide-react';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { ScrollArea } from '../ui/scroll-area';
import { ParsedFile, ZipParseResult } from '../../services/zip-parser';

interface FileTreeProps {
  parseResult: ZipParseResult;
  onFileSelect?: (file: ParsedFile) => void;
  selectedFile?: ParsedFile;
  showPreview?: boolean;
  maxPreviewLines?: number;
}

interface FileNodeProps {
  file: ParsedFile;
  level: number;
  onSelect?: (file: ParsedFile) => void;
  isSelected?: boolean;
  selectedFile?: ParsedFile;
  expandedDirs: Set<string>;
  onToggleDir: (path: string) => void;
}

const FileNode: React.FC<FileNodeProps> = ({ 
  file, 
  level, 
  onSelect, 
  isSelected, 
  selectedFile,
  expandedDirs, 
  onToggleDir 
}) => {
  const isExpanded = expandedDirs.has(file.path);
  
  const getFileIcon = (file: ParsedFile) => {
    if (file.type === 'directory') {
      return isExpanded ? <FolderOpen className="w-4 h-4" /> : <Folder className="w-4 h-4" />;
    }
    
    if (file.isCode) {
      return <Code className="w-4 h-4 text-blue-500" />;
    }
    
    const ext = file.extension.toLowerCase();
    if (['.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico'].includes(ext)) {
      return <Image className="w-4 h-4 text-green-500" />;
    }
    
    if (['.zip', '.tar', '.gz', '.rar'].includes(ext)) {
      return <Archive className="w-4 h-4 text-orange-500" />;
    }
    
    return <FileText className="w-4 h-4 text-gray-500" />;
  };

  const getFileColor = (file: ParsedFile) => {
    if (file.type === 'directory') return 'text-blue-600';
    if (file.isCode) return 'text-green-700';
    return 'text-gray-700';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  if (file.type === 'directory') {
    return (
      <div>
        <Collapsible open={isExpanded} onOpenChange={() => onToggleDir(file.path)}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className={`w-full justify-start h-6 px-2 hover:bg-gray-100 ${
                isSelected ? 'bg-blue-50 border-l-2 border-blue-500' : ''
              }`}
              style={{ paddingLeft: `${level * 16 + 8}px` }}
              onClick={(e) => {
                e.stopPropagation();
                onToggleDir(file.path);
                onSelect?.(file);
              }}
            >
              {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              {getFileIcon(file)}
              <span className={`ml-1 text-sm ${getFileColor(file)}`}>{file.name}</span>
              {file.children && (
                <Badge variant="outline" className="ml-auto text-xs">
                  {file.children.length}
                </Badge>
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            {file.children?.map((child) => (
              <FileNode
                key={child.path}
                file={child}
                level={level + 1}
                onSelect={onSelect}
                isSelected={selectedFile?.path === child.path}
                selectedFile={selectedFile}
                expandedDirs={expandedDirs}
                onToggleDir={onToggleDir}
              />
            ))}
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  }

  return (
    <Button
      variant="ghost"
      className={`w-full justify-start h-6 px-2 hover:bg-gray-100 ${
        isSelected ? 'bg-blue-50 border-l-2 border-blue-500' : ''
      }`}
      style={{ paddingLeft: `${level * 16 + 8}px` }}
      onClick={() => onSelect?.(file)}
    >
      {getFileIcon(file)}
      <span className={`ml-1 text-sm ${getFileColor(file)} flex-1 text-left truncate`}>
        {file.name}
      </span>
      <span className="text-xs text-gray-400 ml-2">{formatFileSize(file.size)}</span>
      {file.language && (
        <Badge variant="outline" className="ml-1 text-xs">
          {file.language}
        </Badge>
      )}
    </Button>
  );
};

const CodePreview: React.FC<{ file: ParsedFile; maxLines?: number }> = ({ 
  file, 
  maxLines = 50 
}) => {
  const lines = file.content.split('\n');
  const displayLines = maxLines ? lines.slice(0, maxLines) : lines;
  const truncated = maxLines && lines.length > maxLines;
  const displayContent = displayLines.join('\n');

  // Map file language to syntax highlighter language
  const getSyntaxLanguage = (language?: string): string => {
    if (!language) return 'text';
    
    const languageMap: Record<string, string> = {
      'javascript': 'javascript',
      'typescript': 'typescript',
      'python': 'python',
      'java': 'java',
      'csharp': 'csharp',
      'cpp': 'cpp',
      'c': 'c',
      'ruby': 'ruby',
      'php': 'php',
      'go': 'go',
      'rust': 'rust',
      'swift': 'swift',
      'kotlin': 'kotlin',
      'scala': 'scala',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'sass': 'sass',
      'json': 'json',
      'yaml': 'yaml',
      'xml': 'xml',
      'sql': 'sql',
      'bash': 'bash',
      'powershell': 'powershell',
      'dockerfile': 'docker',
      'markdown': 'markdown'
    };
    
    return languageMap[language] || 'text';
  };

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-900">{file.name}</h4>
        <div className="flex items-center space-x-2">
          {file.language && (
            <Badge variant="outline" className="text-xs">
              {file.language}
            </Badge>
          )}
          <span className="text-xs text-gray-500">
            {lines.length} lines, {(file.size / 1024).toFixed(1)}KB
          </span>
        </div>
      </div>
      <div className="relative">
        {/* Fallback to plain text to prevent white screen issues */}
        {file.isCode && file.language ? (
          <div className="bg-gray-50 rounded-lg border">
            <div className="p-2 bg-gray-100 border-b text-xs text-gray-600 font-medium">
              {file.language} • {lines.length} lines
            </div>
            <pre className="p-3 text-xs overflow-x-auto max-h-96 overflow-y-auto">
              <code className="text-gray-800 whitespace-pre-wrap">
                {displayContent}
              </code>
            </pre>
          </div>
        ) : (
          <pre className="bg-gray-50 rounded-lg p-3 text-xs overflow-x-auto border max-h-96 overflow-y-auto">
            <code className="text-gray-800">
              {displayContent}
            </code>
          </pre>
        )}
        {truncated && (
          <div className="mt-2 px-3 py-2 bg-gray-100 rounded text-xs text-gray-500 italic border-t">
            ... {lines.length - maxLines} more lines (showing first {maxLines} lines)
          </div>
        )}
      </div>
    </div>
  );
};

export const FileTree: React.FC<FileTreeProps> = ({ 
  parseResult, 
  onFileSelect, 
  selectedFile,
  showPreview = true,
  maxPreviewLines = 50
}) => {
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'tree' | 'list'>('tree');
  const [filterLanguage, setFilterLanguage] = useState<string>('');

  const handleToggleDir = (path: string) => {
    const newExpanded = new Set(expandedDirs);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedDirs(newExpanded);
  };

  const handleExpandAll = () => {
    const allDirs = new Set<string>();
    const addDirs = (files: ParsedFile[]) => {
      files.forEach(file => {
        if (file.type === 'directory') {
          allDirs.add(file.path);
          if (file.children) {
            addDirs(file.children);
          }
        }
      });
    };
    addDirs(parseResult.fileTree);
    setExpandedDirs(allDirs);
  };

  const handleCollapseAll = () => {
    setExpandedDirs(new Set());
  };

  const getFilteredFiles = () => {
    if (!filterLanguage) return parseResult.supportedFiles;
    return parseResult.supportedFiles.filter(file => file.language === filterLanguage);
  };

  const uniqueLanguages = [...new Set(
    parseResult.supportedFiles
      .map(f => f.language)
      .filter(Boolean)
  )].sort();

  const stats = {
    totalFiles: parseResult.totalFiles,
    codeFiles: parseResult.totalCodeFiles,
    totalSize: parseResult.totalSize,
    skippedFiles: parseResult.skippedFiles.length
  };

  return (
    <div className="space-y-4">
      {/* Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="p-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{stats.totalFiles}</div>
            <div className="text-xs text-gray-500">Total Files</div>
          </div>
        </Card>
        <Card className="p-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{stats.codeFiles}</div>
            <div className="text-xs text-gray-500">Code Files</div>
          </div>
        </Card>
        <Card className="p-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{uniqueLanguages.length}</div>
            <div className="text-xs text-gray-500">Languages</div>
          </div>
        </Card>
        <Card className="p-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{stats.skippedFiles}</div>
            <div className="text-xs text-gray-500">Skipped</div>
          </div>
        </Card>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Button
            variant={viewMode === 'tree' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('tree')}
          >
            Tree View
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            List View
          </Button>
        </div>
        
        <div className="flex items-center space-x-2">
          {uniqueLanguages.length > 0 && (
            <select
              value={filterLanguage}
              onChange={(e) => setFilterLanguage(e.target.value)}
              className="px-2 py-1 text-xs border rounded"
            >
              <option value="">All Languages</option>
              {uniqueLanguages.map(lang => (
                <option key={lang} value={lang}>{lang}</option>
              ))}
            </select>
          )}
          
          {viewMode === 'tree' && (
            <>
              <Button size="sm" variant="outline" onClick={handleExpandAll}>
                Expand All
              </Button>
              <Button size="sm" variant="outline" onClick={handleCollapseAll}>
                Collapse All
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* File Tree/List */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">
              {viewMode === 'tree' ? 'Project Structure' : 'File List'}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-96">
              <div className="p-3 space-y-1">
                {viewMode === 'tree' ? (
                  parseResult.fileTree.map((file) => (
                    <FileNode
                      key={file.path}
                      file={file}
                      level={0}
                      onSelect={onFileSelect}
                      isSelected={selectedFile?.path === file.path}
                      selectedFile={selectedFile}
                      expandedDirs={expandedDirs}
                      onToggleDir={handleToggleDir}
                    />
                  ))
                ) : (
                  getFilteredFiles().map((file) => (
                    <FileNode
                      key={file.path}
                      file={file}
                      level={0}
                      onSelect={onFileSelect}
                      isSelected={selectedFile?.path === file.path}
                      selectedFile={selectedFile}
                      expandedDirs={expandedDirs}
                      onToggleDir={handleToggleDir}
                    />
                  ))
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* File Preview */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">File Preview</CardTitle>
          </CardHeader>
          <CardContent>
            {selectedFile && selectedFile.type === 'file' ? (
              showPreview ? (
                <CodePreview file={selectedFile} maxLines={maxPreviewLines} />
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <File className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                  <p>Preview disabled</p>
                </div>
              )
            ) : (
              <div className="text-center py-8 text-gray-500">
                <File className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                <p>Select a file to preview</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Languages Summary */}
      {uniqueLanguages.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Detected Languages</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {uniqueLanguages.map((language) => {
                const count = parseResult.supportedFiles.filter(f => f.language === language).length;
                return (
                  <Badge key={language} variant="outline" className="text-xs">
                    {language} ({count})
                  </Badge>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default FileTree;