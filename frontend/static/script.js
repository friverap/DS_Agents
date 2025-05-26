// Global voice indicator functions
function showVoiceIndicator(message, timeout = 0) {
  // Verificar que el DOM esté cargado
  if (typeof document === 'undefined') {
    console.warn('DOM not available for showing voice indicator');
    return;
  }
  
  let voiceIndicator = document.getElementById('voice-indicator');
  
  if (!voiceIndicator) {
    voiceIndicator = document.createElement('div');
    voiceIndicator.id = 'voice-indicator';
    voiceIndicator.className = 'voice-indicator';
    document.body.appendChild(voiceIndicator);
  }
  
  voiceIndicator.textContent = message;
  voiceIndicator.style.display = 'block';
  voiceIndicator.style.opacity = '1';
  
  if (timeout > 0) {
    setTimeout(() => {
      hideVoiceIndicator();
    }, timeout);
  }
}

function hideVoiceIndicator() {
  // Verificar que el DOM esté cargado
  if (typeof document === 'undefined') {
    console.warn('DOM not available for hiding voice indicator');
    return;
  }
  
  const voiceIndicator = document.getElementById('voice-indicator');
  if (voiceIndicator) {
    voiceIndicator.style.opacity = '0';
    setTimeout(() => {
      if (voiceIndicator.parentNode) {
        voiceIndicator.style.display = 'none';
      }
    }, 300);
  }
}

// Global variables
let conversations = [];
let currentConversation = null;
let folders = [];
let currentFolder = null;
let models = [];  // Will be loaded from the backend API
let selectedModel = null;
let darkMode = localStorage.getItem('darkMode') === 'true';
let voiceSocket = null;
let isListening = false;
let userProfile = {
  name: localStorage.getItem('userName') || "User Name",
  avatarUrl: localStorage.getItem('userAvatarUrl') || "https://via.placeholder.com/40",
  voiceAutoRespond: localStorage.getItem('voiceAutoRespond') === 'true',
  voiceLanguage: localStorage.getItem('voiceLanguage') || 'en-US',
  rating: parseInt(localStorage.getItem('userRating') || 0)
};

// Python code execution with Pyodide
let pyodide = null;
let pyodideReady = false;

// Initialize Pyodide
async function initializePyodide() {
  if (pyodideReady) return pyodide;
  
  try {
    showVoiceIndicator('Loading Python environment...', 0);
    
    // Load Pyodide from CDN
    pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/"
    });
    
    // Install only essential packages that are available
    await pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
    
    // Set up matplotlib to use HTML5 canvas backend
    pyodide.runPython(`
      import matplotlib
      matplotlib.use('module://matplotlib_pyodide.html5_canvas_backend')
      import matplotlib.pyplot as plt
      import numpy as np
      import pandas as pd
      import io
      import base64
      import json
      
      # Configure matplotlib for better plots
      plt.style.use('default')
      plt.rcParams['figure.figsize'] = (10, 6)
      plt.rcParams['figure.dpi'] = 100
      
      def save_plot_as_base64():
          """Save current matplotlib plot as base64 string"""
          buf = io.BytesIO()
          plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
          buf.seek(0)
          img_str = base64.b64encode(buf.read()).decode('utf-8')
          plt.close()  # Close the figure to free memory
          return img_str
      
      def execute_and_capture(code):
          """Execute code and capture both output and plots"""
          import sys
          from io import StringIO
          
          # Capture stdout
          old_stdout = sys.stdout
          sys.stdout = captured_output = StringIO()
          
          result = {
              'output': '',
              'plots': [],
              'error': None,
              'variables': {}
          }
          
          try:
              # Execute the code
              exec(code, globals())
              
              # Check if there are any matplotlib figures
              if plt.get_fignums():
                  for fig_num in plt.get_fignums():
                      plt.figure(fig_num)
                      plot_data = save_plot_as_base64()
                      result['plots'].append(plot_data)
              
              # Get captured output
              result['output'] = captured_output.getvalue()
              
              # Get some variable information (optional)
              # This could be expanded to show variable states
              
          except Exception as e:
              result['error'] = str(e)
          finally:
              sys.stdout = old_stdout
              
          return json.dumps(result)
    `);
    
    pyodideReady = true;
    hideVoiceIndicator();
    showVoiceIndicator('Python environment ready!', 2000);
    
    return pyodide;
  } catch (error) {
    console.error('Error initializing Pyodide:', error);
    hideVoiceIndicator();
    showVoiceIndicator('Error loading Python environment', 3000);
    throw error;
  }
}

// Execute Python code
async function executePythonCode(code) {
  try {
    if (!pyodideReady) {
      await initializePyodide();
    }
    
    // Clean the code before execution
    const cleanedCode = cleanPythonCode(code);
    
    // Execute the code and get results
    const resultJson = pyodide.runPython(`execute_and_capture("""${cleanedCode}""")`);
    const result = JSON.parse(resultJson);
    
    return result;
  } catch (error) {
    console.error('Error executing Python code:', error);
    return {
      output: '',
      plots: [],
      error: error.message,
      variables: {}
    };
  }
}

// Clean Python code to fix formatting issues
function cleanPythonCode(code) {
  if (!code || typeof code !== 'string') {
    return code;
  }
  
  let cleaned = code;
  
  // Step 1: Normalize line endings
  cleaned = cleaned.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  
  // Step 2: Fix broken operators FIRST (before line processing)
  cleaned = cleaned.replace(/\/\s*\/\s*/g, '//');
  cleaned = cleaned.replace(/<\s*=/g, '<=');
  cleaned = cleaned.replace(/>\s*=/g, '>=');
  cleaned = cleaned.replace(/=\s*=/g, '==');
  cleaned = cleaned.replace(/!\s*=/g, '!=');
  cleaned = cleaned.replace(/\*\s*\*/g, '**');
  
  // Step 3: Fix broken file paths with spaces in UUIDs
  cleaned = cleaned.replace(/uploads\s*\/\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*\.\s*csv/g, 
                           'uploads/$1-$2-$3-$4-$5.csv');
  
  // Step 4: Fix broken string literals and f-strings
  cleaned = cleaned.replace(/f\s*'/g, "f'");
  cleaned = cleaned.replace(/f\s*"/g, 'f"');
  
  // Step 5: Fix broken comment patterns
  cleaned = cleaned.replace(/=\s*=\s*=/g, '===');
  
  // Step 6: CRITICAL - Fix concatenated statements by adding line breaks
  // This is the main issue causing the formatting problems
  
  // Fix concatenated print statements
  cleaned = cleaned.replace(/(\))([a-zA-Z_#])/g, '$1\n$2');
  
  // Fix concatenated assignments
  cleaned = cleaned.replace(/([a-zA-Z0-9_\])])([a-zA-Z_][a-zA-Z0-9_]*\s*=)/g, '$1\n$2');
  
  // Fix concatenated imports
  cleaned = cleaned.replace(/(import\s+[a-zA-Z0-9_.,\s]+)([a-zA-Z_])/g, '$1\n$2');
  
  // Fix concatenated function calls
  cleaned = cleaned.replace(/(\))([a-zA-Z_][a-zA-Z0-9_]*\s*\()/g, '$1\n$2');
  
  // Fix concatenated if/for/while statements
  cleaned = cleaned.replace(/([a-zA-Z0-9_\])])(if|for|while|def|class)\s/g, '$1\n$2 ');
  
  // Step 7: Process line by line for detailed fixes
  const lines = cleaned.split('\n');
  const fixedLines = [];
  
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    const trimmed = line.trim();
    
    // Skip empty lines
    if (!trimmed) {
      fixedLines.push(line);
      continue;
    }
    
    // Get original indentation
    const indentMatch = line.match(/^(\s*)/);
    const indent = indentMatch ? indentMatch[1] : '';
    
    // Fix missing # in comments
    if (!trimmed.startsWith('#') && 
        !trimmed.includes('=') && 
        !trimmed.includes('(') && 
        !trimmed.includes('[') &&
        !trimmed.startsWith('import') &&
        !trimmed.startsWith('from')) {
      
      // Check if this looks like a comment
      const commentPatterns = [
        /^(Load|Display|Check|Create|Basic|Get|Missing|Statistical|Distribution|Box|Correlation|Bar|Unique|Additional|Summary)/i,
        /^\d+\.\s*[A-Za-z]/,
        /^[A-Z][a-z]+\s+(the|of|for|with|and|data|information|values|plots|columns)/i,
        /^(Configuración|Crear|Calcular|Mostrar|Análisis|Gráfico|Importar|Generar|Visualizar)/i
      ];
      
      let isComment = false;
      for (const pattern of commentPatterns) {
        if (pattern.test(trimmed)) {
          isComment = true;
          break;
        }
      }
      
      if (isComment) {
        line = indent + '# ' + trimmed;
      }
    }
    
    // Fix spacing around operators and function calls
    if (!line.trim().startsWith('#')) {
      // Fix assignment operators (but not comparison)
      line = line.replace(/\s*=\s*(?!=)/g, ' = ');
      
      // Fix comma spacing
      line = line.replace(/,\s*/g, ', ');
      
      // Fix parentheses spacing
      line = line.replace(/\(\s+/g, '(');
      line = line.replace(/\s+\)/g, ')');
      
      // Fix bracket spacing
      line = line.replace(/\[\s+/g, '[');
      line = line.replace(/\s+\]/g, ']');
      
      // Fix string quotes spacing
      line = line.replace(/'\s+/g, "'");
      line = line.replace(/\s+'/g, "'");
      line = line.replace(/"\s+/g, '"');
      line = line.replace(/\s+"/g, '"');
      
      // Fix method calls with spaces
      line = line.replace(/\.\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, '.$1(');
      
      // Fix keyword arguments
      line = line.replace(/=\s*True/g, '=True');
      line = line.replace(/=\s*False/g, '=False');
      line = line.replace(/=\s*None/g, '=None');
      
      // Fix numeric values
      line = line.replace(/=\s*(\d+\.?\d*)/g, '=$1');
      
      // Fix string values in assignments
      line = line.replace(/=\s*'/g, "='");
      line = line.replace(/=\s*"/g, '="');
    }
    
    fixedLines.push(line);
  }
  
  cleaned = fixedLines.join('\n');
  
  // Step 8: Final cleanup - ensure no concatenated statements remain
  
  // Split by common statement endings and ensure line breaks
  const statementEndings = [
    /(\))\s*([a-zA-Z_#])/g,  // After function calls
    /(\.show\(\))\s*([a-zA-Z_#])/g,  // After matplotlib show()
    /(\.head\(\))\s*([a-zA-Z_#])/g,  // After pandas head()
    /(\.sum\(\))\s*([a-zA-Z_#])/g,   // After sum()
    /(\.describe\(\))\s*([a-zA-Z_#])/g,  // After describe()
  ];
  
  statementEndings.forEach(pattern => {
    cleaned = cleaned.replace(pattern, '$1\n$2');
  });
  
  // Step 9: Fix specific matplotlib/pandas concatenation issues
  cleaned = cleaned.replace(/(plt\.show\(\))\s*([a-zA-Z_#])/g, '$1\n\n$2');
  cleaned = cleaned.replace(/(plt\.tight_layout\(\))\s*([a-zA-Z_#])/g, '$1\n$2');
  
  // Step 10: Ensure proper spacing around print statements
  cleaned = cleaned.replace(/print\s*\(\s*/g, 'print(');
  cleaned = cleaned.replace(/\s*\)\s*([a-zA-Z_#])/g, ')\n$1');
  
  // Step 11: Final line-by-line validation and cleanup
  const finalLines = cleaned.split('\n');
  const validatedLines = [];
  
  for (let i = 0; i < finalLines.length; i++) {
    let line = finalLines[i];
    
    // Remove any remaining multiple spaces (except in strings)
    if (!line.trim().startsWith('#')) {
      // Only fix spacing outside of strings
      let inString = false;
      let stringChar = null;
      let result = '';
      
      for (let j = 0; j < line.length; j++) {
        const char = line[j];
        
        if ((char === '"' || char === "'") && (j === 0 || line[j-1] !== '\\')) {
          if (!inString) {
            inString = true;
            stringChar = char;
          } else if (char === stringChar) {
            inString = false;
            stringChar = null;
          }
        }
        
        if (!inString && char === ' ' && j > 0 && line[j-1] === ' ') {
          // Skip multiple spaces outside strings
          continue;
        }
        
        result += char;
      }
      
      line = result;
    }
    
    validatedLines.push(line);
  }
  
  return validatedLines.join('\n');
}

// Create a code execution interface with editing capabilities
function createCodeExecutionInterface(code, language = 'python') {
  const codeContainer = document.createElement('div');
  codeContainer.className = 'code-execution-container';
  
  // Clean the code for display
  const displayCode = language === 'python' ? cleanPythonCode(code) : code;
  
  const codeHeader = document.createElement('div');
  codeHeader.className = 'code-header';
  codeHeader.innerHTML = `
    <span class="code-language">${language}</span>
    <div class="code-actions">
      <button class="edit-btn" onclick="toggleCodeEdit(this)">
        <i class="fas fa-edit"></i> Editar
      </button>
      <button class="execute-btn" onclick="executeCodeBlock(this)">
        <i class="fas fa-play"></i> Ejecutar
      </button>
      <button class="copy-btn" onclick="copyCodeToClipboard(this)">
        <i class="fas fa-copy"></i> Copiar
      </button>
    </div>
  `;
  
  // Create both display and edit versions
  const codeDisplayBlock = document.createElement('pre');
  codeDisplayBlock.className = 'code-block code-display';
  const codeDisplayElement = document.createElement('code');
  codeDisplayElement.textContent = displayCode;
  codeDisplayBlock.appendChild(codeDisplayElement);
  
  // Create editable textarea (initially hidden)
  const codeEditBlock = document.createElement('div');
  codeEditBlock.className = 'code-edit-container';
  codeEditBlock.style.display = 'none';
  
  const codeTextarea = document.createElement('textarea');
  codeTextarea.className = 'code-edit-textarea';
  codeTextarea.value = displayCode;
  codeTextarea.spellcheck = false;
  
  // Add edit controls
  const editControls = document.createElement('div');
  editControls.className = 'edit-controls';
  editControls.innerHTML = `
    <button class="save-edit-btn" onclick="saveCodeEdit(this)">
      <i class="fas fa-save"></i> Guardar Cambios
    </button>
    <button class="cancel-edit-btn" onclick="cancelCodeEdit(this)">
      <i class="fas fa-times"></i> Cancelar
    </button>
    <button class="format-btn" onclick="formatCode(this)">
      <i class="fas fa-magic"></i> Auto-formatear
    </button>
  `;
  
  codeEditBlock.appendChild(codeTextarea);
  codeEditBlock.appendChild(editControls);
  
  const outputContainer = document.createElement('div');
  outputContainer.className = 'code-output';
  outputContainer.style.display = 'none';
  
  codeContainer.appendChild(codeHeader);
  codeContainer.appendChild(codeDisplayBlock);
  codeContainer.appendChild(codeEditBlock);
  codeContainer.appendChild(outputContainer);
  
  return codeContainer;
}

// Toggle between display and edit mode
function toggleCodeEdit(button) {
  const container = button.closest('.code-execution-container');
  const displayBlock = container.querySelector('.code-display');
  const editBlock = container.querySelector('.code-edit-container');
  const editBtn = container.querySelector('.edit-btn');
  
  if (editBlock.style.display === 'none') {
    // Switch to edit mode
    displayBlock.style.display = 'none';
    editBlock.style.display = 'block';
    editBtn.innerHTML = '<i class="fas fa-eye"></i> Ver';
    
    // Focus on textarea and adjust height
    const textarea = editBlock.querySelector('.code-edit-textarea');
    textarea.focus();
    adjustTextareaHeight(textarea);
  } else {
    // Switch to display mode
    displayBlock.style.display = 'block';
    editBlock.style.display = 'none';
    editBtn.innerHTML = '<i class="fas fa-edit"></i> Editar';
  }
}

// Save code edits
function saveCodeEdit(button) {
  const container = button.closest('.code-execution-container');
  const displayBlock = container.querySelector('.code-display');
  const editBlock = container.querySelector('.code-edit-container');
  const textarea = editBlock.querySelector('.code-edit-textarea');
  const codeElement = displayBlock.querySelector('code');
  
  // Update the display with edited code
  codeElement.textContent = textarea.value;
  
  // Switch back to display mode
  displayBlock.style.display = 'block';
  editBlock.style.display = 'none';
  
  const editBtn = container.querySelector('.edit-btn');
  editBtn.innerHTML = '<i class="fas fa-edit"></i> Editar';
  
  // Show success message
  showVoiceIndicator('Código actualizado exitosamente', 2000);
}

// Cancel code edits
function cancelCodeEdit(button) {
  const container = button.closest('.code-execution-container');
  const displayBlock = container.querySelector('.code-display');
  const editBlock = container.querySelector('.code-edit-container');
  const textarea = editBlock.querySelector('.code-edit-textarea');
  const codeElement = displayBlock.querySelector('code');
  
  // Restore original code
  textarea.value = codeElement.textContent;
  
  // Switch back to display mode
  displayBlock.style.display = 'block';
  editBlock.style.display = 'none';
  
  const editBtn = container.querySelector('.edit-btn');
  editBtn.innerHTML = '<i class="fas fa-edit"></i> Editar';
}

// Enhanced format code function with backend validation
async function formatCode(button) {
  const container = button.closest('.code-execution-container');
  const textarea = container.querySelector('.code-edit-textarea');
  
  // Show loading state
  const originalText = button.innerHTML;
  button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Formateando...';
  button.disabled = true;
  
  try {
    // First, try backend formatting
    const response = await callBackendAPI('/api/format/python', 'POST', {
      code: textarea.value
    });
    
    if (response.formatted_code) {
      textarea.value = response.formatted_code;
      
      // Show changes made
      if (response.changes_made && response.changes_made.length > 0) {
        showVoiceIndicator(`Código formateado: ${response.changes_made.join(', ')}`, 3000);
      } else {
        showVoiceIndicator('Código formateado automáticamente', 2000);
      }
    } else {
      // Fallback to local formatting
      const cleanedCode = cleanPythonCode(textarea.value);
      textarea.value = cleanedCode;
      showVoiceIndicator('Código formateado localmente', 2000);
    }
  } catch (error) {
    console.error('Backend formatting failed, using local formatting:', error);
    // Fallback to local formatting
    const cleanedCode = cleanPythonCode(textarea.value);
    textarea.value = cleanedCode;
    showVoiceIndicator('Código formateado localmente', 2000);
  }
  
  // Adjust height after formatting
  adjustTextareaHeight(textarea);
  
  // Check for remaining issues
  await checkCodeFormattingWithBackend(textarea);
  
  // Reset button
  button.innerHTML = originalText;
  button.disabled = false;
}

// Enhanced code formatting check with backend validation
async function checkCodeFormattingWithBackend(textarea) {
  const code = textarea.value;
  const container = textarea.closest('.code-execution-container');
  
  // Remove existing warnings
  const existingWarning = container.querySelector('.formatting-warning');
  if (existingWarning) {
    existingWarning.remove();
  }
  
  if (!code.trim()) {
    return;
  }
  
  try {
    // Use backend validation
    const response = await callBackendAPI('/api/validate/python', 'POST', {
      code: code
    });
    
    const allIssues = [
      ...response.errors.map(error => `❌ ${error}`),
      ...response.warnings.map(warning => `⚠️ ${warning}`)
    ];
    
    if (allIssues.length > 0) {
      showFormattingWarningWithSuggestions(container, allIssues, response.suggestions, !response.valid);
    }
  } catch (error) {
    console.error('Backend validation failed, using local validation:', error);
    // Fallback to local validation
    checkCodeFormatting(textarea);
  }
}

// Enhanced formatting warning with suggestions
function showFormattingWarningWithSuggestions(container, issues, suggestions = [], hasErrors = false) {
  const warning = document.createElement('div');
  warning.className = hasErrors ? 'formatting-error' : 'formatting-warning';
  
  const headerClass = hasErrors ? 'error-header' : 'warning-header';
  const headerIcon = hasErrors ? 'fas fa-times-circle' : 'fas fa-exclamation-triangle';
  const headerText = hasErrors ? 'Errores de código detectados' : 'Problemas de formateo detectados';
  
  warning.innerHTML = `
    <div class="${headerClass}">
      <i class="${headerIcon}"></i>
      <span>${headerText}</span>
      <div class="warning-actions">
        ${!hasErrors ? `
          <button class="auto-fix-btn" onclick="autoFixFormattingWithBackend(this)">
            <i class="fas fa-magic"></i> Auto-corregir
          </button>
        ` : ''}
        <button class="validate-btn" onclick="validateCodeWithBackend(this)">
          <i class="fas fa-check"></i> Validar
        </button>
      </div>
    </div>
    <ul class="warning-list">
      ${issues.map(issue => `<li>${issue}</li>`).join('')}
    </ul>
    ${suggestions.length > 0 ? `
      <div class="suggestions-section">
        <strong>💡 Sugerencias:</strong>
        <ul class="suggestions-list">
          ${suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
        </ul>
      </div>
    ` : ''}
  `;
  
  // Insert warning before edit controls
  const editControls = container.querySelector('.edit-controls');
  editControls.parentNode.insertBefore(warning, editControls);
}

// Auto-fix formatting with backend support
async function autoFixFormattingWithBackend(button) {
  const container = button.closest('.code-execution-container');
  const textarea = container.querySelector('.code-edit-textarea');
  
  // Show loading state
  const originalText = button.innerHTML;
  button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Corrigiendo...';
  button.disabled = true;
  
  try {
    // Try backend formatting first
    const response = await callBackendAPI('/api/format/python', 'POST', {
      code: textarea.value
    });
    
    if (response.formatted_code) {
      textarea.value = response.formatted_code;
      adjustTextareaHeight(textarea);
      
      // Show changes made
      if (response.changes_made && response.changes_made.length > 0) {
        showVoiceIndicator(`Correcciones aplicadas: ${response.changes_made.join(', ')}`, 3000);
      } else {
        showVoiceIndicator('Código corregido automáticamente', 2000);
      }
    } else {
      // Fallback to local formatting
      autoFixFormatting(button);
      return;
    }
  } catch (error) {
    console.error('Backend auto-fix failed, using local fix:', error);
    // Fallback to local formatting
    autoFixFormatting(button);
    return;
  }
  
  // Remove warning and re-validate
  const warning = container.querySelector('.formatting-warning, .formatting-error');
  if (warning) {
    warning.remove();
  }
  
  // Re-validate after fixing
  setTimeout(() => {
    checkCodeFormattingWithBackend(textarea);
  }, 500);
  
  // Reset button
  button.innerHTML = originalText;
  button.disabled = false;
}

// Validate code with backend
async function validateCodeWithBackend(button) {
  const container = button.closest('.code-execution-container');
  const textarea = container.querySelector('.code-edit-textarea');
  
  // Show loading state
  const originalText = button.innerHTML;
  button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validando...';
  button.disabled = true;
  
  try {
    const response = await callBackendAPI('/api/validate/python', 'POST', {
      code: textarea.value
    });
    
    // Remove existing warnings
    const existingWarning = container.querySelector('.formatting-warning, .formatting-error');
    if (existingWarning) {
      existingWarning.remove();
    }
    
    if (response.valid && response.warnings.length === 0) {
      // Show success message
      const successDiv = document.createElement('div');
      successDiv.className = 'validation-success';
      successDiv.innerHTML = `
        <div class="success-header">
          <i class="fas fa-check-circle"></i>
          <span>✅ Código válido y bien formateado</span>
        </div>
      `;
      
      const editControls = container.querySelector('.edit-controls');
      editControls.parentNode.insertBefore(successDiv, editControls);
      
      // Remove success message after 3 seconds
      setTimeout(() => {
        if (successDiv.parentNode) {
          successDiv.remove();
        }
      }, 3000);
      
      showVoiceIndicator('Código validado exitosamente', 2000);
    } else {
      // Show validation results
      const allIssues = [
        ...response.errors.map(error => `❌ ${error}`),
        ...response.warnings.map(warning => `⚠️ ${warning}`)
      ];
      
      if (allIssues.length > 0) {
        showFormattingWarningWithSuggestions(container, allIssues, response.suggestions, !response.valid);
      }
    }
  } catch (error) {
    console.error('Backend validation failed:', error);
    showVoiceIndicator('Error al validar código', 2000);
  }
  
  // Reset button
  button.innerHTML = originalText;
  button.disabled = false;
}

// Update the input event listener to use backend validation
document.addEventListener('input', (e) => {
  if (e.target.classList.contains('code-edit-textarea')) {
    adjustTextareaHeight(e.target);
    
    // Debounce the validation to avoid too many API calls
    clearTimeout(e.target.validationTimeout);
    e.target.validationTimeout = setTimeout(() => {
      checkCodeFormattingWithBackend(e.target);
    }, 1000); // Wait 1 second after user stops typing
  }
});

// Copy code to clipboard
function copyCodeToClipboard(button) {
  const container = button.closest('.code-execution-container');
  const codeElement = container.querySelector('.code-display code');
  const code = codeElement.textContent;
  
  navigator.clipboard.writeText(code).then(() => {
    showVoiceIndicator('Código copiado al portapapeles', 2000);
    
    // Visual feedback
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i> Copiado';
    setTimeout(() => {
      button.innerHTML = originalText;
    }, 2000);
  }).catch(err => {
    console.error('Error copying to clipboard:', err);
    showVoiceIndicator('Error al copiar código', 2000);
  });
}

// Adjust textarea height based on content
function adjustTextareaHeight(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.max(200, textarea.scrollHeight) + 'px';
}

// Add event listener for textarea auto-resize
document.addEventListener('input', (e) => {
  if (e.target.classList.contains('code-edit-textarea')) {
    adjustTextareaHeight(e.target);
    // Check for formatting issues in real-time
    checkCodeFormatting(e.target);
  }
});

// Execute code from a code block (updated to use current code)
async function executeCodeBlock(button) {
  const container = button.closest('.code-execution-container');
  const outputContainer = container.querySelector('.code-output');
  
  // Get code from display or edit mode
  let code;
  const editBlock = container.querySelector('.code-edit-container');
  if (editBlock.style.display !== 'none') {
    // If in edit mode, use textarea content
    code = editBlock.querySelector('.code-edit-textarea').value;
  } else {
    // If in display mode, use display content
    code = container.querySelector('.code-display code').textContent;
  }
  
  // Show loading state
  button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Ejecutando...';
  button.disabled = true;
  
  try {
    const result = await executePythonCode(code);
    
    // Clear previous output
    outputContainer.innerHTML = '';
    outputContainer.style.display = 'block';
    
    // Show error if any
    if (result.error) {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'code-error';
      errorDiv.innerHTML = `<strong>Error:</strong> ${result.error}`;
      outputContainer.appendChild(errorDiv);
    }
    
    // Show text output if any
    if (result.output && result.output.trim()) {
      const outputDiv = document.createElement('div');
      outputDiv.className = 'code-text-output';
      outputDiv.innerHTML = `<pre>${result.output}</pre>`;
      outputContainer.appendChild(outputDiv);
    }
    
    // Show plots if any
    if (result.plots && result.plots.length > 0) {
      result.plots.forEach((plotData, index) => {
        const plotDiv = document.createElement('div');
        plotDiv.className = 'code-plot-output';
        plotDiv.innerHTML = `
          <img src="data:image/png;base64,${plotData}" alt="Plot ${index + 1}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        `;
        outputContainer.appendChild(plotDiv);
      });
    }
    
    // If no output or plots, show a success message
    if (!result.error && !result.output.trim() && (!result.plots || result.plots.length === 0)) {
      const successDiv = document.createElement('div');
      successDiv.className = 'code-success';
      successDiv.innerHTML = '<i class="fas fa-check"></i> Código ejecutado exitosamente';
      outputContainer.appendChild(successDiv);
    }
    
  } catch (error) {
    console.error('Error executing code:', error);
    outputContainer.innerHTML = `<div class="code-error"><strong>Error:</strong> ${error.message}</div>`;
    outputContainer.style.display = 'block';
  } finally {
    // Reset button
    button.innerHTML = '<i class="fas fa-play"></i> Ejecutar';
    button.disabled = false;
  }
}

// Detect and enhance code blocks in AI responses
function enhanceCodeBlocks(messageElement) {
  console.log('Enhancing code blocks in message element:', messageElement);
  const codeBlocks = messageElement.querySelectorAll('pre code');
  console.log('Found code blocks:', codeBlocks.length);
  
  codeBlocks.forEach((codeElement, index) => {
    const code = codeElement.textContent;
    console.log(`Code block ${index}:`, code.substring(0, 100));
    
    // Check if it's Python code (simple heuristic)
    if (isPythonCode(code)) {
      console.log(`Code block ${index} detected as Python, enhancing...`);
      const preElement = codeElement.parentElement;
      const executionInterface = createCodeExecutionInterface(code, 'python');
      
      // Replace the original pre element with our enhanced version
      preElement.parentNode.replaceChild(executionInterface, preElement);
      console.log(`Code block ${index} enhanced successfully`);
    } else {
      console.log(`Code block ${index} not detected as Python`);
    }
  });
}

// Simple heuristic to detect Python code
function isPythonCode(code) {
  const pythonKeywords = [
    'import ', 'from ', 'def ', 'class ', 'if __name__',
    'plt.', 'np.', 'pd.', 'matplotlib', 'numpy', 'pandas',
    'print(', 'plt.show()', 'plt.plot(', 'plt.figure(',
    'sns.', 'seaborn', 'plt.subplot', 'plt.title',
    'plt.xlabel', 'plt.ylabel', 'plt.grid', 'plt.legend'
  ];
  
  // Check for Python keywords
  const hasKeywords = pythonKeywords.some(keyword => code.includes(keyword));
  
  // Check for Python syntax patterns
  const hasPythonSyntax = /^\s*(import|from|def|class|if|for|while|try|with)\s/.test(code) ||
                          /plt\.\w+\(/.test(code) ||
                          /np\.\w+\(/.test(code) ||
                          /pd\.\w+\(/.test(code) ||
                          /sns\.\w+\(/.test(code);
  
  console.log('Code detection:', { code: code.substring(0, 100), hasKeywords, hasPythonSyntax });
  
  return hasKeywords || hasPythonSyntax;
}

// Initialize Pyodide when the page loads
document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM loaded, initializing Pyodide...');
  
  // Initialize Pyodide immediately
  initializePyodide().then(() => {
    console.log('Pyodide initialized successfully');
  }).catch(error => {
    console.error('Failed to initialize Pyodide:', error);
    // Show user-friendly message
    showVoiceIndicator('Python environment failed to load. Code execution may not work.', 5000);
  });
});

// Utility function to call the backend API
async function callBackendAPI(endpoint, method = 'GET', data = null) {
  try {
    console.log(`API Request to ${endpoint}:`, { method, data });
    
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      }
    };
    
    if (data) {
      options.body = JSON.stringify(data);
    }
    
    const response = await fetch(endpoint, options);
    
    // Log the status and headers for debugging
    console.log(`API Response status: ${response.status}`, {
      statusText: response.statusText,
      headers: Object.fromEntries([...response.headers])
    });
    
    if (!response.ok) {
      // Try to get error details from the response
      let errorDetail;
      try {
        errorDetail = await response.text();
        console.error(`API error response body: ${errorDetail}`);
      } catch (textError) {
        console.error('Failed to extract error details:', textError);
      }
      
      throw new Error(`API request failed with status ${response.status}: ${errorDetail || response.statusText}`);
    }
    
    // Get response data
    const responseData = await response.json();
    console.log(`API Response data from ${endpoint}:`, responseData);
    return responseData;
  } catch (error) {
    console.error(`Error calling API ${endpoint}:`, error);
    throw error;
  }
}

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
  const chatArea = document.getElementById('chatArea') || document.querySelector('.chat-area');
  const inputField = document.getElementById('messageInput') || document.querySelector('.input-area input');
  const sendButton = document.getElementById('sendBtn') || document.querySelector('.send-btn');
  const newChatButton = document.querySelector('.new-chat-btn');
  const modelSelector = document.querySelector('.model-selector');
  const themeToggle = document.getElementById('theme-toggle');
  const createFolderBtn = document.getElementById('create-folder-btn');
  const folderModal = document.getElementById('folder-modal');
  const saveFolderBtn = document.getElementById('save-folder-btn');
  const closeModal = document.querySelector('.close-modal');
  const folderNameInput = document.getElementById('folder-name-input');
  const foldersList = document.getElementById('folders-list');
  const emptyFolderMessage = document.getElementById('empty-folder-message');
  const chatsList = document.getElementById('chats-list');
  const emptyChatMessage = document.getElementById('empty-chat-message');
  const globeBtn = document.querySelector('.globe-btn');
  const micBtn = document.querySelector('.mic-btn');
  const uploadBtn = document.querySelector('.upload-btn');
  const codeBtn = document.querySelector('.code-btn');
  const settingsBtn = document.getElementById('settings-btn');
  const settingsModal = document.getElementById('settings-modal');
  const saveSettingsBtn = document.getElementById('save-settings-btn');
  const displayNameInput = document.getElementById('display-name');
  const userAvatarEl = document.getElementById('user-avatar');
  const userNameEl = document.getElementById('user-name');
  const settingsAvatarEl = document.getElementById('settings-avatar');
  const avatarUploadBtn = document.querySelector('.avatar-upload-btn');
  const avatarInput = document.getElementById('avatar-input');
  const voiceAutoRespondCheckbox = document.getElementById('voice-autorespond');
  const voiceLanguageSelect = document.getElementById('voice-language');
  const userPointsEl = document.querySelector('.user-points');
  const menuButton = document.querySelector('.user-profile .fa-ellipsis-v');
  const ratingModal = document.getElementById('rating-modal');
  const ratingSlider = document.getElementById('rating-slider');
  const ratingDisplay = document.getElementById('rating-display');
  const feedbackText = document.getElementById('feedback-text');
  const submitRatingBtn = document.getElementById('submit-rating-btn');
  
  // Add a variable to track focus mode state
  let isFocusMode = false;
  
  // Initialize the app
  initApp();
  
  // Set initial theme
  if (darkMode) {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
  } else {
    document.documentElement.setAttribute('data-theme', 'light');
    themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
  }
  
  // Event listeners
  sendButton.addEventListener('click', sendMessage);
  inputField.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  });
  
  newChatButton.addEventListener('click', createNewChat);
  
  modelSelector.addEventListener('click', toggleModelSelection);
  
  // Theme toggle event listener
  themeToggle.addEventListener('click', () => {
    darkMode = !darkMode;
    if (darkMode) {
      document.documentElement.setAttribute('data-theme', 'dark');
      themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    localStorage.setItem('darkMode', darkMode);
  });
  
  // Web search button event listener
  globeBtn.addEventListener('click', () => {
    const query = inputField.value.trim();
    if (query) {
      performWebSearch(query);
    } else {
      // Create search modal if input is empty
      createSearchModal();
    }
  });
  
  // File upload button event listener
  uploadBtn.addEventListener('click', triggerFileUpload);
  
  // Settings button event listener
  settingsBtn.addEventListener('click', openSettingsModal);
  
  // Save settings button event listener
  saveSettingsBtn.addEventListener('click', saveSettings);
  
  // Avatar upload button event listener
  avatarUploadBtn.addEventListener('click', () => {
    avatarInput.click();
  });
  
  // Avatar input change event listener
  avatarInput.addEventListener('change', handleAvatarUpload);
  
  // Folder creation event listeners
  createFolderBtn.addEventListener('click', () => {
    folderModal.style.display = 'flex';
    folderNameInput.focus();
  });
  
  // Close modal event listener for all modals
  document.querySelectorAll('.close-modal').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const modal = e.target.closest('.modal');
      if (modal) {
        modal.style.display = 'none';
      }
    });
  });
  
  // Click outside modal to close
  window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
      e.target.style.display = 'none';
    }
  });
  
  saveFolderBtn.addEventListener('click', createFolder);
  
  folderNameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      createFolder();
    }
  });

  // Menu button event listener (three vertical dots)
  menuButton.addEventListener('click', openRatingModal);
  
  // Rating slider event listener
  ratingSlider.addEventListener('input', updateRatingDisplay);
  
  // Submit rating button event listener
  submitRatingBtn.addEventListener('click', submitRating);

  // Create hidden file input for document uploads
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.style.display = 'none';
  fileInput.accept = '.txt,.pdf,.csv,.json,.md,.py,.js,.html,.css,.ipynb,.R,.xlsx,.docx';
  fileInput.multiple = true;
  document.body.appendChild(fileInput);
  
  // Handle file selection
  fileInput.addEventListener('change', handleFileUpload);
  
  // Code button (Focus Mode) event listener
  codeBtn.addEventListener('click', toggleFocusMode);
  
  // Initialize the application
  function initApp() {
    // Load folders and conversations from localStorage
    loadDataFromStorage();
    
    // Initialize user profile
    updateUserProfile();
    
    // Fetch available models from backend
    fetchAvailableModels().then(() => {
      // After models are loaded, set the selected model
      if (models.length > 0) {
        const storedModelId = localStorage.getItem('selectedModel');
        if (storedModelId) {
          const storedModel = models.find(m => m.id === storedModelId);
          selectedModel = storedModel || models[0];
        } else {
          selectedModel = models[0];
        }
      
        // Update UI
        if (modelSelector) {
          const modelSpan = modelSelector.querySelector('span');
          if (modelSpan) {
            modelSpan.textContent = selectedModel.name;
          }
        }
      }
    
      // Display chat history in sidebar
      displayChatHistory();
    
      // Create initial conversation if none exists
      if (conversations.length === 0) {
        createNewChat();
      } else {
        // Load the most recent conversation
        loadConversation(conversations[conversations.length - 1]);
      }
    });
    
    // Display folders
    displayFolders();
    
    // Check voice assistant status
    checkVoiceAssistantStatus();
  }
  
  // Update user profile in the UI
  function updateUserProfile() {
    userNameEl.textContent = userProfile.name;
    userAvatarEl.src = userProfile.avatarUrl;
    userPointsEl.textContent = userProfile.rating + " points";
    
    // Update settings modal values
    if (settingsAvatarEl) {
      settingsAvatarEl.src = userProfile.avatarUrl;
    }
    if (displayNameInput) {
      displayNameInput.value = userProfile.name;
    }
    if (voiceAutoRespondCheckbox) {
      voiceAutoRespondCheckbox.checked = userProfile.voiceAutoRespond;
    }
    if (voiceLanguageSelect) {
      voiceLanguageSelect.value = userProfile.voiceLanguage;
    }
  }
  
  // Open rating modal
  function openRatingModal() {
    // Set current rating value
    ratingSlider.value = userProfile.rating;
    ratingDisplay.textContent = userProfile.rating;
    
    // Clear feedback text
    feedbackText.value = '';
    
    // Show modal
    ratingModal.style.display = 'flex';
  }
  
  // Update rating display when slider moves
  function updateRatingDisplay() {
    ratingDisplay.textContent = ratingSlider.value;
  }
  
  // Submit rating and feedback
  function submitRating() {
    // Update user profile with new rating
    userProfile.rating = parseInt(ratingSlider.value);
    
    // Save to localStorage
    localStorage.setItem('userRating', userProfile.rating);
    
    // Update UI
    updateUserProfile();
    
    // Get feedback text
    const feedback = feedbackText.value.trim();
    
    // Send feedback via email if provided
    if (feedback) {
      sendFeedbackEmail(feedback, userProfile.rating);
    }
    
    // Close modal
    ratingModal.style.display = 'none';
    
    // Show confirmation message
    showVoiceIndicator('Thank you for your feedback!', 2000);
  }
  
  // Send feedback email using mailto or a server endpoint
  function sendFeedbackEmail(feedback, rating) {
    // Option 1: Open mailto link (simple but requires user's email client)
    const subject = encodeURIComponent(`DSAgency Feedback: Rating ${rating}/100`);
    const body = encodeURIComponent(`Rating: ${rating}/100\n\nFeedback:\n${feedback}\n\nUser: ${userProfile.name}`);
    
    // Create a hidden anchor element
    const mailtoLink = document.createElement('a');
    mailtoLink.href = `mailto:friverap.fr@gmail.com?subject=${subject}&body=${body}`;
    mailtoLink.style.display = 'none';
    document.body.appendChild(mailtoLink);
    
    // Click the link and then remove it
    mailtoLink.click();
    document.body.removeChild(mailtoLink);
    
    // Option 2: Send via server-side endpoint (use this if you have a backend)
    // This would be implemented if you have a server-side API
    /*
    fetch('/api/feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        rating: rating,
        feedback: feedback,
        user: userProfile.name,
        email: 'friverap.fr@gmail.com'
      })
    })
    .then(response => response.json())
    .then(data => console.log('Feedback sent:', data))
    .catch(error => console.error('Error sending feedback:', error));
    */
  }
  
  // Open settings modal
  function openSettingsModal() {
    // Update settings with current values
    displayNameInput.value = userProfile.name;
    settingsAvatarEl.src = userProfile.avatarUrl;
    voiceAutoRespondCheckbox.checked = userProfile.voiceAutoRespond;
    voiceLanguageSelect.value = userProfile.voiceLanguage;
    
    // Show modal
    settingsModal.style.display = 'flex';
  }
  
  // Save settings
  function saveSettings() {
    // Update user profile
    userProfile.name = displayNameInput.value.trim() || "User Name";
    userProfile.voiceAutoRespond = voiceAutoRespondCheckbox.checked;
    userProfile.voiceLanguage = voiceLanguageSelect.value;
    
    // Save to localStorage
    localStorage.setItem('userName', userProfile.name);
    localStorage.setItem('userAvatarUrl', userProfile.avatarUrl);
    localStorage.setItem('voiceAutoRespond', userProfile.voiceAutoRespond);
    localStorage.setItem('voiceLanguage', userProfile.voiceLanguage);
    
    // Update UI
    updateUserProfile();
    
    // Close modal
    settingsModal.style.display = 'none';
    
    // Show confirmation message
    showVoiceIndicator('Settings saved successfully', 2000);
  }
  
  // Handle avatar upload
  function handleAvatarUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const avatarDataUrl = e.target.result;
      
      // Update user profile and UI
      userProfile.avatarUrl = avatarDataUrl;
      settingsAvatarEl.src = avatarDataUrl;
      
      // No need to save settings here as that happens when the Save button is clicked
    };
    reader.readAsDataURL(file);
  }
  
  // Trigger file upload dialog
  function triggerFileUpload() {
    fileInput.click();
  }
  
  // Handle file upload
  async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    showVoiceIndicator('Uploading files...', 0);
    
    // Create a FormData object to send the files
    const formData = new FormData();
    
    // Add each file to the form data
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    
    try {
      // Upload the files to the backend
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Failed to upload files');
      }
      
      const data = await response.json();
      
      // Store the uploaded file paths for later use in chat requests
      if (data.uploaded_files && data.uploaded_files.length > 0) {
        // Store the file paths in the conversation object
        if (!currentConversation.uploaded_files) {
          currentConversation.uploaded_files = [];
        }
        
        // Add all uploaded file paths to the conversation
        data.uploaded_files.forEach(fileInfo => {
          currentConversation.uploaded_files.push(fileInfo.path);
        });
        
        console.log('Stored file paths:', currentConversation.uploaded_files);
      }
      
      // Create a message showing the uploaded files
      let fileListHtml = '## Uploaded Files\n\n';
      for (let i = 0; i < files.length; i++) {
        fileListHtml += `- ${files[i].name} (${formatFileSize(files[i].size)})\n`;
      }
      
      // Add file list to conversation
      const fileListMessage = {
        id: Date.now().toString(),
        role: 'user',
        content: fileListHtml,
        timestamp: new Date().toISOString()
      };
      
      if (currentConversation) {
        currentConversation.messages.push(fileListMessage);
        displayMessage(fileListMessage);
        
        // Create AI response with file analysis
        const aiResponseMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `I've received your files. You can ask me questions about them or request an analysis.`,
          timestamp: new Date().toISOString()
        };
        
        currentConversation.messages.push(aiResponseMessage);
        displayMessage(aiResponseMessage);
        saveDataToStorage();
      }
      
      hideVoiceIndicator();
      
    } catch (error) {
      console.error('Error uploading files:', error);
      showVoiceIndicator('Error uploading files', 3000);
    }
    
    // Reset the file input
    fileInput.value = '';
  }
  
  // Format file size in a human-readable format
  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  // Load data from local storage
  function loadDataFromStorage() {
    const storedFolders = localStorage.getItem('folders');
    if (storedFolders) {
      folders = JSON.parse(storedFolders);
    }
    
    const storedConversations = localStorage.getItem('conversations');
    if (storedConversations) {
      conversations = JSON.parse(storedConversations);
    }
    
    const storedSelectedModel = localStorage.getItem('selectedModel');
    if (storedSelectedModel) {
      const model = models.find(m => m.id === storedSelectedModel);
      if (model) {
        selectedModel = model;
      }
    }
  }
  
  // Save data to local storage
  function saveDataToStorage() {
    localStorage.setItem('folders', JSON.stringify(folders));
    localStorage.setItem('conversations', JSON.stringify(conversations));
    localStorage.setItem('selectedModel', selectedModel.id);
  }
  
  // Create a new folder
  function createFolder() {
    const folderName = folderNameInput.value.trim();
    
    if (folderName === '') return;
    
    const newFolder = {
      id: Date.now().toString(),
      name: folderName,
      conversations: []
    };
    
    folders.push(newFolder);
    saveDataToStorage();
    
    // Reset and close modal
    folderNameInput.value = '';
    folderModal.style.display = 'none';
    
    // Display folders
    displayFolders();
  }
  
  // Display folders in the sidebar
  function displayFolders() {
    foldersList.innerHTML = '';
    
    if (folders.length === 0) {
      emptyFolderMessage.style.display = 'flex';
      return;
    }
    
    emptyFolderMessage.style.display = 'none';
    
    folders.forEach(folder => {
      const folderElement = document.createElement('div');
      folderElement.className = 'folder';
      if (currentFolder && folder.id === currentFolder.id) {
        folderElement.classList.add('active');
      }
      
      folderElement.innerHTML = `
        <i class="fas fa-folder"></i>
        <div class="folder-name">${folder.name}</div>
        <div class="folder-options">
          <i class="fas fa-ellipsis-v"></i>
        </div>
      `;
      
      folderElement.addEventListener('click', () => {
        currentFolder = folder;
        
        // Update UI to show folder is selected
        document.querySelectorAll('.folder').forEach(el => el.classList.remove('active'));
        folderElement.classList.add('active');
        
        // Load conversations for this folder
        if (folder.conversations.length > 0) {
          const conversationId = folder.conversations[0];
          const conversation = conversations.find(c => c.id === conversationId);
          if (conversation) {
            loadConversation(conversation);
          }
        } else {
          // Create a new conversation in this folder
          createNewChat(folder.id);
        }
      });
      
      foldersList.appendChild(folderElement);
    });
  }
  
  // Create a new chat conversation
  function createNewChat(folderId = null) {
    const newConversation = {
      id: Date.now().toString(),
      title: 'New Conversation',
      messages: [],
      folderId: folderId
    };
    
    conversations.push(newConversation);
    
    // If creating in a folder, add to folder's conversations
    if (folderId) {
      const folder = folders.find(f => f.id === folderId);
      if (folder) {
        folder.conversations.push(newConversation.id);
      }
    }
    
    saveDataToStorage();
    loadConversation(newConversation);
    
    // Clear welcome message
    chatArea.innerHTML = '';
  }
  
  // Load a specific conversation
  function loadConversation(conversation) {
    currentConversation = conversation;
    
    // Clear chat area
    chatArea.innerHTML = '';
    
    // Display conversation messages
    conversation.messages.forEach(message => {
      displayMessage(message);
    });
    
    // If no messages, show empty state
    if (conversation.messages.length === 0) {
      showWelcomeMessage();
    }
    
    // Update folder selection if needed
    if (conversation.folderId) {
      const folder = folders.find(f => f.id === conversation.folderId);
      if (folder && (!currentFolder || currentFolder.id !== folder.id)) {
        currentFolder = folder;
        document.querySelectorAll('.folder').forEach(el => {
          if (el.querySelector('.folder-name').textContent === folder.name) {
            el.classList.add('active');
          } else {
            el.classList.remove('active');
          }
        });
      }
    }
  }
  
  // Fetch available models from the backend
  async function fetchAvailableModels() {
    try {
      const response = await callBackendAPI('/api/models/providers');
      
      // Reset models array
      models = [];
      
      // Parse provider data from the response
      if (response && response.providers) {
        // The response has a 'providers' array with provider objects
        response.providers.forEach(provider => {
          const providerName = provider.name;
          const modelsList = provider.models || [];
          
          // Add each model from this provider
          modelsList.forEach(model => {
            models.push({
              id: `${providerName}/${model.id}`,
              name: model.name,
              description: model.description,
              provider: providerName,
              isDefault: false
            });
          });
        });
        
        // Set the current model as default if available
        if (response.current_model && response.current_provider) {
          const currentModelId = `${response.current_provider}/${response.current_model}`;
          const currentModel = models.find(m => m.id === currentModelId);
          if (currentModel) {
            currentModel.isDefault = true;
          }
        }
      }
      
      console.log('Loaded models:', models);
      
      // If no models were loaded, use a default model
      if (models.length === 0) {
        console.warn('No models returned from the backend, adding a default model');
        models.push({
          id: 'default/model',
          name: 'Default Model',
          description: 'Default model when none are available',
          provider: 'default'
        });
      }
      
      return models;
    } catch (error) {
      console.error('Error fetching available models:', error);
      
      // Add a default model in case of error
      models = [{
        id: 'default/model',
        name: 'Default Model',
        description: 'Default model when connection fails',
        provider: 'default'
      }];
      
      return models;
    }
  }
  
  // Send a message
  function sendMessage() {
    // Get DOM elements with error checking
    const messageInput = document.getElementById('messageInput');
    const chatArea = document.getElementById('chatArea') || document.querySelector('.chat-area');
    
    if (!messageInput) {
      console.error('Message input element not found');
      return;
    }
    
    if (!chatArea) {
      console.error('Chat area element not found');
      return;
    }
    
    const message = messageInput.value.trim();
    
    if (message === '') return;
    
    // Check if we have a current conversation
    if (!currentConversation) {
      console.log('No current conversation, creating new one...');
      createNewChat();
    }
    
    // Check if we have a selected model
    if (!selectedModel) {
      console.log('No selected model, using default...');
      if (models.length > 0) {
        selectedModel = models[0];
      } else {
        // Create a default model if none exist
        selectedModel = {
          id: 'anthropic/claude-3-5-sonnet-20241022',
          name: 'Claude 3.5 Sonnet',
          provider: 'anthropic'
        };
      }
    }
    
    // Create user message object
    const userMessage = {
      id: Date.now().toString(),
      content: message,
      role: 'user',
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    // Add to current conversation
    currentConversation.messages.push(userMessage);
    
    // Auto-save to storage
    saveDataToStorage();
    
    // Display the message
    displayMessage(userMessage);
    
    // Clear input
    messageInput.value = '';
    
    // Auto-scroll to bottom
    chatArea.scrollTop = chatArea.scrollHeight;
    
    // Send to AI
    sendAIResponse(message);
    
    // Update chat history in sidebar
    displayChatHistory();
  }
  
  // Determine if a message is a simple conversational query
  function isSimpleConversationalQuery(message) {
    const messageLower = message.toLowerCase().trim();
    
    // Check if it explicitly mentions data analysis, EDA, or file analysis
    const dataAnalysisKeywords = [
      "eda", "exploratory data analysis", "análisis exploratorio",
      "dataset", "data", "csv", "excel", "analyze", "analiza", "analizar",
      "visualization", "visualización", "visualizar", "plot", "graph", "gráfico",
      "modelo", "model", "prediction", "predicción", "machine learning",
      "execute", "run", "ejecuta", "corre", "correlation", "correlación",
      "statistics", "estadísticas", "distribution", "distribución",
      "missing values", "valores faltantes", "outliers", "atípicos",
      "feature", "características", "target", "objetivo", "training",
      "entrenamiento", "test", "prueba", "validation", "validación",
      "regression", "regresión", "classification", "clasificación",
      "clustering", "agrupamiento", "preprocessing", "preprocesamiento"
    ];
    
    // If it mentions data analysis keywords, it's NOT a simple query
    if (dataAnalysisKeywords.some(keyword => messageLower.includes(keyword))) {
      console.log('Detected data analysis query, using complex endpoint');
      return false;
    }
    
    // Check if there are uploaded files in the current conversation
    if (currentConversation && currentConversation.uploaded_files && 
        currentConversation.uploaded_files.length > 0) {
      console.log('Files uploaded, treating as complex query');
      return false;
    }
    
    // Simple greeting patterns
    const greetingPatterns = [
      /^(hi|hello|hola|hey|good morning|good afternoon|good evening)/,
      /^(how are you|como estas|what's up|que tal)/,
      /^(my name is|me llamo|i am|soy)/,
      /^(what is|que es|what are|que son)/,
      /^(can you|puedes|could you|podrias)/,
      /^(tell me about|cuentame sobre|explain|explica)/,
      /^(what agents|que agentes|what can you do|que puedes hacer)/,
      /^(help|ayuda|how do i|como hago)/,
    ];
    
    // Check if it matches simple conversational patterns
    for (const pattern of greetingPatterns) {
      if (pattern.test(messageLower)) {
        console.log('Detected simple conversational query');
        return true;
      }
    }
    
    // Check if it's a simple question without data analysis intent
    const simpleQuestionKeywords = [
      "what is", "que es", "what are", "que son", "how do i", "como hago",
      "can you explain", "puedes explicar", "tell me", "cuentame",
      "what agents", "que agentes", "available", "disponible"
    ];
    
    // If it's a short message with simple question keywords, it's probably conversational
    if (message.split(' ').length < 15 && simpleQuestionKeywords.some(keyword => messageLower.includes(keyword))) {
      console.log('Detected simple question');
      return true;
    }
    
    // If none of the above, consider it complex if it's longer than 10 words
    const isComplex = message.split(' ').length > 10;
    console.log(`Query classification: ${isComplex ? 'complex' : 'simple'}`);
    return !isComplex;
  }
  
  // Send AI response using the backend API
  async function sendAIResponse(userMessage) {
    // Create a processing message element
    const processing = displayProcessingMessage();
    
    try {
      // Check if we have a selected model
      if (!selectedModel) {
        console.error('No selected model available');
        throw new Error('No AI model selected. Please select a model first.');
      }
      
      // Extract provider and model name from model.id (format: "provider/model")
      const modelParts = selectedModel.id.split('/');
      if (modelParts.length < 2) {
        console.error('Invalid model ID format:', selectedModel.id);
        throw new Error('Invalid model configuration. Please select a different model.');
      }
      
      const [provider, modelName] = modelParts;
      
      // Get the most recent uploaded file path if available
      const filePath = currentConversation.uploaded_files && 
                       currentConversation.uploaded_files.length > 0 ? 
                       currentConversation.uploaded_files[currentConversation.uploaded_files.length - 1] : 
                       null;
      
      // Determine if this is a simple conversational query
      const isSimpleQuery = isSimpleConversationalQuery(userMessage);
      
      // Choose endpoint based on query type
      const endpoint = isSimpleQuery ? '/api/chat/simple' : '/api/chat';
      
      console.log(`Using ${endpoint} for message: "${userMessage.substring(0, 50)}..."`);
      console.log(`Model: ${provider}/${modelName}`);
      
      // Make API call to the backend using the utility function
      const data = await callBackendAPI(endpoint, 'POST', {
        message: userMessage,
        session_id: currentConversation.id,  // Use session_id instead of conversation_id
        model_provider: provider,
        model_name: modelName,
        file_path: filePath // Pass the file path to the chat endpoint
      });
      
      // Log the complete response for debugging
      console.log('Backend API response:', data);
      
      // Remove processing message
      const chatArea = document.getElementById('chatArea') || document.querySelector('.chat-area');
      if (chatArea && processing && processing.parentNode) {
        chatArea.removeChild(processing);
      }
      
      // Create AI response from the real backend response - improved error handling
      let responseContent = "I'm having trouble generating a response. Please try again.";
      
      // Check different possible response formats
      if (data && typeof data === 'object') {
        if (data.response && typeof data.response === 'string' && data.response.trim() !== '') {
          responseContent = data.response;
        } else if (data.content && typeof data.content === 'string' && data.content.trim() !== '') {
          responseContent = data.content;
        } else if (data.message && typeof data.message === 'string' && data.message.trim() !== '') {
          responseContent = data.message;
        } else if (data.text && typeof data.text === 'string' && data.text.trim() !== '') {
          responseContent = data.text;
        } else if (typeof data === 'string' && data.trim() !== '') {
          responseContent = data;
        }
      }
      
      const response = {
        id: Date.now().toString(),
        content: responseContent,
        role: 'assistant',
        sender: 'ai',
        timestamp: new Date().toISOString(),
        model: selectedModel.id
      };
      
      // Add to conversation
      currentConversation.messages.push(response);
      
      // Auto-save to storage
      saveDataToStorage();
      
      // Display the response
      displayMessage(response);
      
      // Update chat history in sidebar
      displayChatHistory();
      
      // Auto-scroll to bottom
      if (chatArea) {
        chatArea.scrollTop = chatArea.scrollHeight;
      }
      
      // Update conversation title if it's the first message
      if (currentConversation.messages.length === 2) {
        // Extract title from first message
        currentConversation.title = userMessage.length > 20
          ? userMessage.substring(0, 20) + '...'
          : userMessage;
        
        saveDataToStorage();
        displayChatHistory();
      }
    } catch (error) {
      console.error('Error calling AI API:', error);
      
      // Remove processing message
      const chatArea = document.getElementById('chatArea') || document.querySelector('.chat-area');
      if (chatArea && processing && processing.parentNode) {
        chatArea.removeChild(processing);
      }
      
      // Show error message
      const errorResponse = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `❌ Error: ${error.message}\n\nPlease check your connection and try again.`,
        timestamp: new Date().toISOString()
      };
      
      // Add to conversation
      currentConversation.messages.push(errorResponse);
      
      // Auto-save to storage
      saveDataToStorage();
      
      // Display the error response
      displayMessage(errorResponse);
      
      // Auto-scroll to bottom
      if (chatArea) {
        chatArea.scrollTop = chatArea.scrollHeight;
      }
    }
  }
  
  // Display a message in the chat area
  function displayMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.sender || message.role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Check if the content is already HTML (contains div tags)
    if (message.content.includes('<div class="search-results">') || 
        message.content.includes('<h1>') || 
        message.content.includes('<h2>') ||
        message.content.includes('<div class="search-result">')) {
      // Content is already HTML, use it directly
      contentDiv.innerHTML = message.content;
    } else if ((message.sender === 'ai' || message.role === 'assistant') && 
        (message.content.includes('Search Results') || 
         message.content.includes('Web Search Results'))) {
      // Handle legacy search results with improved formatting
      contentDiv.innerHTML = formatSearchResultsToHTML(message.content);
    } else if (message.content.includes('#') || 
               message.content.includes('---') || 
               message.content.includes('[') || 
               message.content.includes('```')) {
      // Convert markdown to HTML for other cases
      contentDiv.innerHTML = convertMarkdownToHtml(message.content);
    } else {
      contentDiv.textContent = message.content;
    }
    
    messageDiv.appendChild(contentDiv);
    chatArea.appendChild(messageDiv);
    
    // Enhance code blocks if this is an AI message
    if (message.sender === 'ai' || message.role === 'assistant') {
      enhanceCodeBlocks(messageDiv);
    }
    
    // Auto-scroll to bottom
    chatArea.scrollTop = chatArea.scrollHeight;
  }
  
  // Format search results directly to HTML
  function formatSearchResultsToHTML(content) {
    if (content.includes('Web Search Results for:')) {
      // Parse existing search results format and convert to cleaner format
      
      // Extract the query
      const queryMatch = content.match(/Web Search Results for: "(.*?)"/);
      const query = queryMatch ? queryMatch[1] : "your search";
      
      // Start building HTML with proper styling
      let html = `<h1>Search Results</h1>`;
      
      // Split by ### to get individual results
      const resultBlocks = content.split('###').slice(1); // skip the first empty part
      
      resultBlocks.forEach((block, index) => {
        // Extract title, link and description
        const titleLinkMatch = block.match(/(\d+)\. \[(.*?)\]\((.*?)\)(.*)/s);
        
        if (titleLinkMatch) {
          const number = titleLinkMatch[1];
          const title = titleLinkMatch[2];
          const url = titleLinkMatch[3];
          const description = titleLinkMatch[4].trim();
          
          // Extract domain
          let domain = '';
          try {
            const urlObj = new URL(url);
            domain = urlObj.hostname.replace('www.', '');
          } catch (e) {
            domain = 'source';
          }
          
          // Add formatted result
          html += `<h3>${number}. ${title}</h3>`;
          html += `<p>${description}</p>`;
          html += `<p><a href="${url}" target="_blank">Read more on ${domain}</a></p>`;
          
          // Add separator except for last item
          if (index < resultBlocks.length - 1) {
            html += `<hr>`;
          }
        }
      });
      
      return html;
    }
    
    // If it's already in our new format, just convert markdown
    return convertMarkdownToHtml(content);
  }
  
  // Format search results in a clean, readable way
  function formatSearchResults(query, results) {
    if (!results || results.length === 0) {
      return `<div class="search-results">
        <h2>Search Results</h2>
        <p>No results found for your search: "${query}"</p>
      </div>`;
    }
    
    let formattedContent = `<div class="search-results">
      <h2>Search Results</h2>
      <p class="search-query">Results for: "${query}"</p>
    `;
    
    // Add each result in a clean format
    results.forEach((result, index) => {
      // Handle both URL-based results and AI-generated responses
      let domain = result.source || 'AI Assistant';
      let url = result.url || '#';
      let title = result.title || `Result ${index + 1}`;
      let content = result.content || result.description || '';
      
      // If it's an AI response, format it differently
      if (result.type === 'ai_response') {
        formattedContent += `
          <div class="search-result ai-response">
            <div class="result-header">
              <h3 class="result-title">${title}</h3>
              <span class="result-source">${domain}</span>
            </div>
            <div class="result-content">
              ${convertMarkdownToHtml(content)}
            </div>
          </div>
        `;
      } else {
        // Regular web search result
        formattedContent += `
          <div class="search-result web-result">
            <div class="result-header">
              <h3 class="result-title">
                <a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>
              </h3>
              <span class="result-source">${domain}</span>
            </div>
            <div class="result-content">
              <p>${content}</p>
              <a href="${url}" target="_blank" rel="noopener noreferrer" class="read-more-link">
                Read more on ${domain}
              </a>
            </div>
          </div>
        `;
      }
      
      // Add separator except for last item
      if (index < results.length - 1) {
        formattedContent += `<hr class="result-separator">`;
      }
    });
    
    formattedContent += `</div>`;
    
    return formattedContent;
  }
  
  // Convert markdown to HTML for better display
  function convertMarkdownToHtml(markdownText) {
    let html = markdownText;
    
    // Store code blocks temporarily to protect them from line break conversion
    const codeBlocks = [];
    let codeBlockIndex = 0;
    
    // Extract and store code blocks
    html = html.replace(/```(\w+)?\n([\s\S]*?)\n```/g, (match, language, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
      codeBlocks[codeBlockIndex] = {
        language: language || '',
        code: code,
        fullMatch: match
      };
      codeBlockIndex++;
      return placeholder;
    });
    
    // Convert headers
    html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');
    html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    
    // Convert horizontal rules
    html = html.replace(/^---$/gm, '<hr>');
    
    // Convert links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Convert bold text
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert italic text
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert line breaks (but not in code blocks)
    html = html.replace(/\n/g, '<br>');
    
    // Restore code blocks with proper formatting
    codeBlocks.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      const formattedCode = `<pre><code class="language-${block.language}">${block.code}</code></pre>`;
      html = html.replace(placeholder, formattedCode);
    });
    
    return html;
  }
  
  // Display processing indicator
  function displayProcessingMessage() {
    const processingDiv = document.createElement('div');
    processingDiv.className = 'message ai-message processing';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    processingDiv.appendChild(contentDiv);
    chatArea.appendChild(processingDiv);
    
    // Auto-scroll to bottom
    chatArea.scrollTop = chatArea.scrollHeight;
    
    return processingDiv;
  }
  
  // Show welcome message
  async function showWelcomeMessage() {
    try {
      // Try to get a welcome message from the backend
      const data = await callBackendAPI('/api/welcome');
      
      const welcomeContainer = document.createElement('div');
      welcomeContainer.className = 'welcome-container';
      
      const heading = document.createElement('h1');
      heading.textContent = data.title || 'Welcome to DSAgency';
      
      const paragraph = document.createElement('p');
      paragraph.textContent = data.message || 'Your AI-powered Data Science Workflow Assistant';
      
      // Create example questions container
      const exampleQuestionsContainer = document.createElement('div');
      exampleQuestionsContainer.className = 'example-questions-container';
      
      // Create example question buttons
      const exampleQuestions = data.example_questions || [
        "How do I clean missing data in a dataset?",
        "What is exploratory data analysis?",
        "Can you analyze this dataset for customer churn patterns?",
        "What agents are available in this system?",
        "How do I build a classification model for customer segmentation?"
      ];
      
      // Add each example question as a button
      exampleQuestions.forEach(question => {
        const questionButton = document.createElement('button');
        questionButton.className = 'example-question-btn';
        questionButton.textContent = question;
        
        // Add click event listener
        questionButton.addEventListener('click', () => {
          // Set the question as the input value
          inputField.value = question;
          
          // Send the message
          sendMessage();
        });
        
        exampleQuestionsContainer.appendChild(questionButton);
      });
      
      welcomeContainer.appendChild(heading);
      welcomeContainer.appendChild(paragraph);
      welcomeContainer.appendChild(exampleQuestionsContainer);
      chatArea.appendChild(welcomeContainer);
    } catch (error) {
      // Fallback to basic welcome if API fails
      console.error('Error getting welcome message:', error);
      
      // Display a static welcome message instead of sending an automatic chat message
      const welcomeContainer = document.createElement('div');
      welcomeContainer.className = 'welcome-container';
      
      const heading = document.createElement('h1');
      heading.textContent = 'Welcome to DSAgency';
      
      const paragraph = document.createElement('p');
      paragraph.textContent = 'Your AI-powered Data Science Workflow Assistant';
      
      // Create example questions container
      const exampleQuestionsContainer = document.createElement('div');
      exampleQuestionsContainer.className = 'example-questions-container';
      
      // Create example question buttons
      const exampleQuestions = [
        "How do I clean missing data in a dataset?",
        "What is exploratory data analysis?",
        "Can you analyze this dataset for customer churn patterns?",
        "What agents are available in this system?",
        "How do I build a classification model for customer segmentation?"
      ];
      
      // Add each example question as a button
      exampleQuestions.forEach(question => {
        const questionButton = document.createElement('button');
        questionButton.className = 'example-question-btn';
        questionButton.textContent = question;
        
        // Add click event listener
        questionButton.addEventListener('click', () => {
          // Set the question as the input value
          inputField.value = question;
          
          // Send the message
          sendMessage();
        });
        
        exampleQuestionsContainer.appendChild(questionButton);
      });
      
      welcomeContainer.appendChild(heading);
      welcomeContainer.appendChild(paragraph);
      welcomeContainer.appendChild(exampleQuestionsContainer);
      chatArea.appendChild(welcomeContainer);
    }
  }
  
  // Toggle model selection dropdown
  function toggleModelSelection() {
    // Check if dropdown already exists
    let dropdown = document.querySelector('.model-dropdown');
    
    if (dropdown) {
      // Remove if exists
      dropdown.remove();
      return;
    }
    
    // Create dropdown
    dropdown = document.createElement('div');
    dropdown.className = 'model-dropdown';
    
    // Group models by provider
    const providers = {};
    models.forEach(model => {
      if (!providers[model.provider]) {
        providers[model.provider] = [];
      }
      providers[model.provider].push(model);
    });
    
    // Add provider sections
    for (const [provider, providerModels] of Object.entries(providers)) {
      const providerSection = document.createElement('div');
      providerSection.className = 'provider-section';
      
      const providerName = document.createElement('div');
      providerName.className = 'provider-name';
      providerName.textContent = formatProviderName(provider);
      providerSection.appendChild(providerName);
      
      // Add models for this provider
      providerModels.forEach(model => {
        const modelOption = document.createElement('div');
        modelOption.className = 'model-option';
        if (model.id === selectedModel.id) {
          modelOption.classList.add('selected');
        }
        
        const modelName = document.createElement('div');
        modelName.className = 'model-name';
        modelName.textContent = model.name;
        
        const modelDesc = document.createElement('div');
        modelDesc.className = 'model-description';
        modelDesc.textContent = model.description;
        
        modelOption.appendChild(modelName);
        modelOption.appendChild(modelDesc);
        
        modelOption.addEventListener('click', async () => {
          // Store previous model for fallback
          const previousModel = selectedModel;
          
          // Update selected model
          selectedModel = model;
          document.querySelector('.model-selector span').textContent = model.name;
          
          // Save to localStorage
          saveDataToStorage();
          
          // Close dropdown
          dropdown.remove();
          
          // Show loading indicator
          const modelText = document.querySelector('.model-selector span');
          const originalText = modelText.textContent;
          modelText.textContent = "Updating...";
          
          try {
            // Extract provider and model name from model.id (format: "provider/model")
            const [provider, modelName] = model.id.split('/');
            
            // Call API to update model configuration on the backend
            await callBackendAPI('/api/models/configure', 'POST', {
              provider: provider,
              model: modelName
            });
            
            // Update model selector text with success
            modelText.textContent = originalText;
            
            // Add a success indicator
            const successIndicator = document.createElement('span');
            successIndicator.className = 'success-indicator';
            successIndicator.textContent = " ✓";
            modelText.appendChild(successIndicator);
            
            // Remove success indicator after 2 seconds
            setTimeout(() => {
              if (modelText.contains(successIndicator)) {
                modelText.removeChild(successIndicator);
              }
            }, 2000);
          } catch (error) {
            console.error('Failed to update model configuration:', error);
            
            // Revert to previous model if the API call fails
            selectedModel = previousModel;
            modelText.textContent = previousModel.name;
            
            // Show error message
            const errorToast = document.createElement('div');
            errorToast.className = 'toast error';
            errorToast.textContent = "Failed to update model. Check browser console for details.";
            document.body.appendChild(errorToast);
            
            // Remove error message after 3 seconds
            setTimeout(() => {
              document.body.removeChild(errorToast);
            }, 3000);
            
            // Save the reverted model to localStorage
            saveDataToStorage();
          }
        });
        
        providerSection.appendChild(modelOption);
      });
      
      dropdown.appendChild(providerSection);
    }
    
    // Get position of model selector
    const modelSelector = document.querySelector('.model-selector');
    const rect = modelSelector.getBoundingClientRect();
    
    // Set dropdown position
    dropdown.style.top = `${rect.bottom}px`;
    dropdown.style.left = `${rect.left}px`;
    
    // Append to document body instead of inside the nav
    document.body.appendChild(dropdown);
    
    // Close dropdown when clicking elsewhere
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.model-dropdown') && !e.target.closest('.model-selector')) {
        if (dropdown.parentNode) {
          dropdown.remove();
        }
      }
    }, { once: true });
  }
  
  // Perform web search
  async function performWebSearch(query) {
    // Show loading message
    const loadingMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: `Web search: ${query}`,
      timestamp: new Date().toISOString()
    };
    
    // Add message to conversation
    if (currentConversation) {
      currentConversation.messages.push(loadingMessage);
      displayMessage(loadingMessage);
      
      // Show typing indicator
      const processing = displayProcessingMessage();
      
      try {
        // Make API request to backend search endpoint using the utility function
        const searchData = await callBackendAPI(`/api/search?q=${encodeURIComponent(query)}`);
        
        // Remove typing indicator
        chatArea.removeChild(processing);
        
        // Format search results using the new formatter
        const resultContent = formatSearchResults(query, searchData.results);
        
        // Create response message
        const responseMessage = {
          id: Date.now().toString(),
          role: 'assistant',
          content: resultContent,
          timestamp: new Date().toISOString()
        };
        
        // Add to conversation and display
        currentConversation.messages.push(responseMessage);
        displayMessage(responseMessage);
        saveDataToStorage();
        displayChatHistory();
        
      } catch (error) {
        console.error('Search error:', error);
        
        // Remove typing indicator
        chatArea.removeChild(processing);
        
        // Try to get an error message from the backend
        let errorContent = "Sorry, I encountered an error while searching. Please try again later.";
        
        try {
          const errorData = await callBackendAPI('/api/error', 'POST', { error: error.message, type: 'search' });
          if (errorData && errorData.message) {
            errorContent = errorData.message;
          }
        } catch (secondaryError) {
          // If that fails, use the default error message
          console.error('Error fetching error message:', secondaryError);
        }
        
        // Show error message
        const errorResponse = {
          id: Date.now().toString(),
          role: 'assistant',
          content: errorContent,
          timestamp: new Date().toISOString()
        };
        
        currentConversation.messages.push(errorResponse);
        displayMessage(errorResponse);
        saveDataToStorage();
        displayChatHistory();
      }
    }
  }
  
  // Create search modal
  function createSearchModal() {
    // Check if modal already exists
    if (document.getElementById('search-modal')) {
      return;
    }
    
    const searchModal = document.createElement('div');
    searchModal.id = 'search-modal';
    searchModal.className = 'modal';
    searchModal.style.display = 'flex';
    
    searchModal.innerHTML = `
      <div class="modal-content">
        <span class="close-search-modal">&times;</span>
        <h3>Web Search</h3>
        <p>Enter your search query below</p>
        <input type="text" id="search-input" placeholder="Search the web...">
        <button id="perform-search-btn">Search</button>
      </div>
    `;
    
    document.body.appendChild(searchModal);
    
    // Focus search input
    const searchInput = document.getElementById('search-input');
    searchInput.focus();
    
    // Add event listeners
    const closeBtn = document.querySelector('.close-search-modal');
    closeBtn.addEventListener('click', () => {
      searchModal.remove();
    });
    
    searchModal.addEventListener('click', (e) => {
      if (e.target === searchModal) {
        searchModal.remove();
      }
    });
    
    const searchBtn = document.getElementById('perform-search-btn');
    searchBtn.addEventListener('click', () => {
      const query = searchInput.value.trim();
      if (query) {
        performWebSearch(query);
        searchModal.remove();
      }
    });
    
    searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        const query = searchInput.value.trim();
        if (query) {
          performWebSearch(query);
          searchModal.remove();
        }
      }
    });
  }
  
  // Initialize WebSocket for voice recognition
  function initVoiceWebSocket() {
    console.log('Initializing WebSocket connection');
    
    // Close existing connection if any
    if (voiceSocket && voiceSocket.readyState !== WebSocket.CLOSED) {
      console.log('Closing existing WebSocket connection');
      voiceSocket.close();
    }
    
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/voice/ws`;
    console.log('Connecting to WebSocket URL:', wsUrl);
    
    try {
      voiceSocket = new WebSocket(wsUrl);
      
      // Connection opened
      voiceSocket.addEventListener('open', (event) => {
        console.log('WebSocket connection established successfully');
        showVoiceIndicator('Voice service connected', 2000);
      });
      
      // Listen for messages
      voiceSocket.addEventListener('message', (event) => {
        console.log('WebSocket message received:', event.data);
        try {
          const data = JSON.parse(event.data);
          handleVoiceSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      });
      
      // Handle connection close
      voiceSocket.addEventListener('close', (event) => {
        console.log('WebSocket connection closed. Code:', event.code, 'Reason:', event.reason);
        
        // Try to reconnect after a delay if the page is still open
        setTimeout(() => {
          if (document.visibilityState !== 'hidden') {
            console.log('Attempting to reconnect WebSocket...');
            initVoiceWebSocket();
          }
        }, 3000);
      });
      
      // Handle errors
      voiceSocket.addEventListener('error', (error) => {
        console.error('WebSocket error:', error);
        showVoiceIndicator('Voice service connection error', 2000);
      });
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      showVoiceIndicator('Failed to connect to voice service', 2000);
    }
  }
  
  // Toggle voice recognition on/off
  function toggleVoiceRecognition() {
    console.log('Toggle voice recognition called');
    console.log('WebSocket state:', voiceSocket ? voiceSocket.readyState : 'no socket');
    console.log('Is listening:', isListening);
    
    if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
      if (isListening) {
        console.log('Stopping listening...');
        // Stop listening
        voiceSocket.send(JSON.stringify({ action: 'stop_listening' }));
        micBtn.classList.remove('active');
      } else {
        console.log('Starting listening...');
        // Start listening
        voiceSocket.send(JSON.stringify({ action: 'start_listening' }));
        micBtn.classList.add('active');
        
        // If in mock mode, show a text input dialog
        checkVoiceAssistantStatus();
      }
    } else {
      console.error('WebSocket is not connected');
      // Try to reconnect
      initVoiceWebSocket();
      
      // Try again after a short delay
      setTimeout(() => {
        if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
          console.log('Retrying after reconnection...');
          toggleVoiceRecognition();
        } else {
          console.error('Failed to reconnect WebSocket');
          showVoiceIndicator('Failed to connect to voice service', 3000);
        }
      }, 1000);
    }
  }
  
  // Check if voice assistant is in mock mode
  async function checkVoiceAssistantStatus() {
    console.log('Checking voice assistant status...');
    try {
      const response = await fetch('/api/voice/status');
      
      if (!response.ok) {
        throw new Error(`Voice status check failed with status ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Voice assistant status:', data);
      
      if (data.mock_mode === true) {
        console.log('Voice assistant is in simulated mode');
        
        // Set isListening to true for the UI
        if (micBtn && micBtn.classList.contains('active')) {
          // Show voice input dialog only if mic button is active
          showMockVoiceInputDialog();
        }
      }
    } catch (error) {
      console.error('Error checking voice assistant status:', error);
      
      // Assume we need to use the input dialog if there's an error
      if (isListening) {
        showVoiceIndicator('Using text input for voice commands', 2000);
        showMockVoiceInputDialog();
      }
    }
  }
  
  // Show a dialog for text input in mock mode
  function showMockVoiceInputDialog() {
    console.log('Showing voice input dialog');
    
    // Remove any existing mock dialog
    const existingDialog = document.getElementById('voice-input-dialog');
    if (existingDialog) {
      existingDialog.remove();
    }
    
    // Create a voice input dialog
    const inputDialog = document.createElement('div');
    inputDialog.id = 'voice-input-dialog';
    inputDialog.style.position = 'fixed';
    inputDialog.style.top = '0';
    inputDialog.style.left = '0';
    inputDialog.style.width = '100%';
    inputDialog.style.height = '100%';
    inputDialog.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    inputDialog.style.zIndex = '9999';
    inputDialog.style.display = 'flex';
    inputDialog.style.justifyContent = 'center';
    inputDialog.style.alignItems = 'center';
    
    inputDialog.innerHTML = `
      <div style="background-color: var(--bg-color); color: var(--text-color); padding: 20px; border-radius: 8px; width: 400px; max-width: 90%; position: relative; z-index: 10000; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);">
        <span id="close-voice-dialog" style="position: absolute; top: 10px; right: 15px; cursor: pointer; font-size: 24px;">&times;</span>
        <h3 style="margin-top: 0; margin-bottom: 15px;">Voice Input</h3>
        <p style="margin-bottom: 15px; color: var(--text-secondary);">Type what you would say:</p>
        <input type="text" id="voice-input-field" style="width: 100%; padding: 10px; margin-bottom: 15px; box-sizing: border-box; border: 1px solid var(--border-color); border-radius: 4px; background-color: var(--input-bg); color: var(--text-color);" placeholder="Type your voice command...">
        <div style="display: flex; justify-content: space-between;">
          <button id="cancel-voice-input" style="padding: 10px 20px; background-color: var(--button-bg); color: var(--button-text); border: none; cursor: pointer; border-radius: 4px;">Cancel</button>
          <button id="send-voice-input" style="padding: 10px 20px; background-color: var(--primary-color); color: white; border: none; cursor: pointer; border-radius: 4px; font-weight: bold;">Send</button>
        </div>
      </div>
    `;
    
    // Append to document body
    document.body.appendChild(inputDialog);
    
    // Focus the input field
    setTimeout(() => {
      const inputField = document.getElementById('voice-input-field');
      if (inputField) {
        inputField.focus();
      }
    }, 100);
    
    // Handle close button
    const closeBtn = document.getElementById('close-voice-dialog');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        inputDialog.remove();
        if (isListening) {
          toggleVoiceRecognition();
        }
      });
    }
    
    // Handle cancel button
    const cancelBtn = document.getElementById('cancel-voice-input');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => {
        inputDialog.remove();
        if (isListening) {
          toggleVoiceRecognition();
        }
      });
    }
    
    // Handle send button
    const sendBtn = document.getElementById('send-voice-input');
    if (sendBtn) {
      sendBtn.addEventListener('click', sendVoiceInput);
    }
    
    // Handle enter key
    const inputField = document.getElementById('voice-input-field');
    if (inputField) {
      inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          sendVoiceInput();
        }
      });
    }
    
    function sendVoiceInput() {
      const inputField = document.getElementById('voice-input-field');
      if (!inputField) return;
      
      const text = inputField.value.trim();
      
      if (text) {
        // Add text directly to input field
        const chatInput = document.querySelector('.input-area input');
        if (chatInput) {
          chatInput.value = text;
          
          // Send the message if auto-respond is enabled
          if (userProfile.voiceAutoRespond) {
            sendMessage();
          }
        }
        
        // Remove the dialog
        inputDialog.remove();
        
        // Stop listening
        if (isListening) {
          toggleVoiceRecognition();
        }
      }
    }
  }
  
  // Handle status messages from the voice assistant
  function handleStatusMessage(data) {
    const status = data.status;
    console.log('Received status message:', status);
    
    switch (status) {
      case 'ready':
        console.log('Voice assistant is ready');
        isListening = false;
        micBtn.classList.remove('active');
        hideVoiceIndicator();
        break;
      case 'listening':
        console.log('Voice assistant is now listening');
        isListening = true;
        micBtn.classList.add('active');
        const listeningIndicator = showVoiceIndicator('Listening...');
        if (listeningIndicator) {
          listeningIndicator.style.backgroundColor = '#4CAF50'; // Green
          listeningIndicator.style.animation = 'pulse 1.5s infinite';
        }
        break;
      case 'processing':
        console.log('Voice assistant is processing speech');
        const processingIndicator = showVoiceIndicator('Processing speech...');
        if (processingIndicator) {
          processingIndicator.style.backgroundColor = '#2196F3'; // Blue
        }
        break;
      case 'not_understood':
        console.log('Voice assistant did not understand');
        isListening = false;
        micBtn.classList.remove('active');
        const notUnderstoodIndicator = showVoiceIndicator('Sorry, I didn\'t understand that', 2000);
        if (notUnderstoodIndicator) {
          notUnderstoodIndicator.style.backgroundColor = '#FF5722'; // Orange-red
        }
        break;
      case 'error':
        console.log('Voice assistant error');
        isListening = false;
        micBtn.classList.remove('active');
        const errorIndicator = showVoiceIndicator('Error processing speech', 2000);
        if (errorIndicator) {
          errorIndicator.style.backgroundColor = '#F44336'; // Red
        }
        break;
      case 'stopped':
        console.log('Voice assistant stopped');
        isListening = false;
        micBtn.classList.remove('active');
        hideVoiceIndicator();
        
        // Remove voice dialog if it exists
        const voiceDialog = document.getElementById('voice-input-dialog');
        if (voiceDialog) {
          voiceDialog.remove();
        }
        break;
      case 'searching':
        console.log('Voice assistant is searching the web');
        const searchingIndicator = showVoiceIndicator('Searching the web...');
        if (searchingIndicator) {
          searchingIndicator.style.backgroundColor = '#FF9800'; // Orange
        }
        break;
      case 'search_error':
        console.log('Voice assistant search error');
        const searchErrorIndicator = showVoiceIndicator('Error performing web search', 2000);
        if (searchErrorIndicator) {
          searchErrorIndicator.style.backgroundColor = '#F44336'; // Red
        }
        break;
      case 'processing_agent':
        console.log('Voice assistant is processing with agent');
        const agentIndicator = showVoiceIndicator('Processing with agent...');
        if (agentIndicator) {
          agentIndicator.style.backgroundColor = '#9C27B0'; // Purple
        }
        break;
      default:
        console.log('Unknown status:', status);
    }
  }
  
  // Handle incoming WebSocket messages
  function handleVoiceSocketMessage(data) {
    console.log('Voice WebSocket message received:', data);
    
    switch (data.type) {
      case 'status':
        handleStatusMessage(data);
        break;
      case 'speech_result':
        console.log('Received speech recognition result:', data.text);
        handleSpeechResult(data);
        break;
      case 'search_results':
        console.log('Received search results');
        handleSearchResults(data);
        break;
      case 'agent_response':
        console.log('Received agent response');
        handleAgentResponse(data);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  }
  
  // Handle speech recognition results
  function handleSpeechResult(data) {
    // Display recognized text in input field
    inputField.value = data.text;
    hideVoiceIndicator();
    
    // If auto-respond is enabled, send the message
    if (userProfile.voiceAutoRespond) {
      sendMessage();
    }
  }
  
  // Handle search results from voice command
  function handleSearchResults(data) {
    // Display the search results in the chat
    const query = data.query;
    const results = data.results;
    
    // Create a search message
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: `Search for: ${query}`,
      timestamp: new Date().toISOString()
    };
    
    // Add message to conversation
    if (currentConversation) {
      currentConversation.messages.push(userMessage);
      displayMessage(userMessage);
      
      // Format search results in a cleaner way
      let resultContent = formatSearchResults(query, results);
      
      // Create response message
      const responseMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: resultContent,
        timestamp: new Date().toISOString()
      };
      
      // Add to conversation and display
      currentConversation.messages.push(responseMessage);
      displayMessage(responseMessage);
      saveDataToStorage();
      displayChatHistory();
    }
    
    hideVoiceIndicator();
  }
  
  // Handle agent responses from voice commands
  function handleAgentResponse(data) {
    const response = data.response;
    
    // Add the agent response to the conversation
    if (currentConversation) {
      // If there's no recent user message, create one with the voice input
      const lastMessage = currentConversation.messages[currentConversation.messages.length - 1];
      
      if (!lastMessage || lastMessage.role !== 'user') {
        // Create a user message from the input field
        const voiceCommand = inputField.value.trim();
        if (voiceCommand) {
          const userMessage = {
            id: Date.now().toString(),
            role: 'user',
            content: voiceCommand,
            timestamp: new Date().toISOString()
          };
          
          currentConversation.messages.push(userMessage);
          displayMessage(userMessage);
        }
      }
      
      // Create and add the AI response
      const responseMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date().toISOString()
      };
      
      currentConversation.messages.push(responseMessage);
      displayMessage(responseMessage);
      saveDataToStorage();
      
      // Clear the input field
      inputField.value = '';
    }
    
    hideVoiceIndicator();
  }
  
  // Add a function to check if microphone is working directly
  function checkMicrophoneAccess() {
    console.log('Checking microphone access...');
    showVoiceIndicator('Checking microphone access...');
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error('Browser does not support getUserMedia API');
      showVoiceIndicator('Browser does not support microphone access', 3000);
      return;
    }
    
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        console.log('Microphone access granted!');
        showVoiceIndicator('Microphone is accessible', 2000);
        
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(track => track.stop());
        
        // Since we verified microphone works, check if we can use the voice assistant
        checkVoiceAssistantStatus();
      })
      .catch(err => {
        console.error('Microphone access error:', err);
        showVoiceIndicator('Microphone access denied: ' + err.message, 3000);
      });
  }
  
  // Add listener to mic button to first check microphone access
  micBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    // First check microphone access directly to see if that's the issue
    if (!isListening) {
      checkMicrophoneAccess();
    } else {
      toggleVoiceRecognition();
    }
  });
  
  // Toggle focus mode function
  function toggleFocusMode() {
    console.log('Toggling focus mode');
    isFocusMode = !isFocusMode;
    
    const appContainer = document.querySelector('.app-container');
    const sidebar = document.querySelector('.sidebar');
    const topNav = document.querySelector('.top-nav');
    const mainContent = document.querySelector('.main-content');
    const inputContainer = document.querySelector('.input-container');
    
    if (isFocusMode) {
      console.log('Enabling focus mode');
      
      // Add focus-mode class to app container
      appContainer.classList.add('focus-mode');
      
      // Hide sidebar
      sidebar.style.display = 'none';
      
      // Hide top navigation
      topNav.style.display = 'none';
      
      // Expand main content and chat area
      mainContent.style.gridColumn = '1 / -1';
      mainContent.style.gridRow = '1 / -1';
      chatArea.style.maxHeight = 'calc(100vh - 80px)';
      
      // Style changes for focus mode
      inputContainer.classList.add('focus-mode');
      codeBtn.classList.add('active');
      
      // Show exit focus mode button
      showVoiceIndicator('Focus mode enabled - click code button to exit', 3000);
    } else {
      console.log('Disabling focus mode');
      
      // Remove focus-mode class from app container
      appContainer.classList.remove('focus-mode');
      
      // Show sidebar
      sidebar.style.display = '';
      
      // Show top navigation
      topNav.style.display = '';
      
      // Reset main content and chat area
      mainContent.style.gridColumn = '';
      mainContent.style.gridRow = '';
      chatArea.style.maxHeight = '';
      
      // Remove focus mode styles
      inputContainer.classList.remove('focus-mode');
      codeBtn.classList.remove('active');
      
      showVoiceIndicator('Focus mode disabled', 2000);
    }
  }
  
  // Display chat history in the sidebar
  function displayChatHistory() {
    if (!chatsList) return;
    
    chatsList.innerHTML = '';
    
    if (conversations.length === 0) {
      if (emptyChatMessage) {
        emptyChatMessage.style.display = 'flex';
      }
      return;
    }
    
    if (emptyChatMessage) {
      emptyChatMessage.style.display = 'none';
    }
    
    // Sort conversations by most recent first (using the timestamp of the last message)
    const sortedConversations = [...conversations].sort((a, b) => {
      const aTime = a.messages.length > 0 ? 
        new Date(a.messages[a.messages.length - 1].timestamp).getTime() : 
        0;
      const bTime = b.messages.length > 0 ? 
        new Date(b.messages[b.messages.length - 1].timestamp).getTime() : 
        0;
      return bTime - aTime;
    });
    
    sortedConversations.forEach(conversation => {
      const chatElement = document.createElement('div');
      chatElement.className = 'chat-item';
      if (currentConversation && conversation.id === currentConversation.id) {
        chatElement.classList.add('active');
      }
      
      // Get preview text from first message or use title
      let previewText = conversation.title || 'New Conversation';
      if (conversation.messages.length > 0) {
        const firstUserMessage = conversation.messages.find(m => m.role === 'user' || m.sender === 'user');
        if (firstUserMessage) {
          previewText = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
        }
      }
      
      // Format date
      let dateText = 'Just now';
      if (conversation.messages.length > 0) {
        const lastMessage = conversation.messages[conversation.messages.length - 1];
        const messageDate = new Date(lastMessage.timestamp);
        const now = new Date();
        const diffTime = Math.abs(now - messageDate);
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) {
          // Today - show time
          dateText = messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (diffDays === 1) {
          dateText = 'Yesterday';
        } else if (diffDays < 7) {
          const options = { weekday: 'short' };
          dateText = messageDate.toLocaleDateString(undefined, options);
        } else {
          const options = { month: 'short', day: 'numeric' };
          dateText = messageDate.toLocaleDateString(undefined, options);
        }
      }
      
      chatElement.innerHTML = `
        <div class="chat-info">
          <div class="chat-title">${previewText}</div>
          <div class="chat-date">${dateText}</div>
        </div>
        <div class="chat-options">
          <i class="fas fa-trash-alt chat-delete-btn"></i>
        </div>
      `;
      
      // Chat click event
      chatElement.addEventListener('click', (e) => {
        // Skip if clicking on delete button
        if (e.target.classList.contains('chat-delete-btn') || 
            e.target.closest('.chat-delete-btn')) {
          return;
        }
        
        // Load the conversation
        loadConversation(conversation);
        
        // Update UI to show chat is selected
        document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
        chatElement.classList.add('active');
      });
      
      // Delete button click event
      const deleteBtn = chatElement.querySelector('.chat-delete-btn');
      if (deleteBtn) {
        deleteBtn.addEventListener('click', (e) => {
          e.stopPropagation(); // Prevent chat from being loaded
          deleteConversation(conversation.id);
        });
      }
      
      chatsList.appendChild(chatElement);
    });
  }
  
  // Delete a conversation
  function deleteConversation(conversationId) {
    // Find conversation index
    const index = conversations.findIndex(c => c.id === conversationId);
    if (index === -1) return;
    
    // Confirm deletion
    if (!confirm('Are you sure you want to delete this conversation?')) {
      return;
    }
    
    // If it's in a folder, remove it from folder's conversations
    const conversation = conversations[index];
    if (conversation.folderId) {
      const folder = folders.find(f => f.id === conversation.folderId);
      if (folder) {
        const convIndex = folder.conversations.indexOf(conversationId);
        if (convIndex !== -1) {
          folder.conversations.splice(convIndex, 1);
        }
      }
    }
    
    // Remove conversation
    conversations.splice(index, 1);
    
    // Save to storage
    saveDataToStorage();
    
    // If current conversation was deleted, load another one
    if (currentConversation && currentConversation.id === conversationId) {
      if (conversations.length > 0) {
        loadConversation(conversations[conversations.length - 1]);
      } else {
        createNewChat();
      }
    }
    
    // Update displayed chats
    displayChatHistory();
  }
});

// Add some CSS to support the JavaScript functionality
document.addEventListener('DOMContentLoaded', () => {
  const style = document.createElement('style');
  style.textContent = `
    .chat-area {
      display: flex;
      flex-direction: column;
      align-items: stretch;
      justify-content: flex-start;
      padding: 20px;
      overflow-y: auto;
    }
    
    .message {
      margin-bottom: 15px;
      max-width: 80%;
      padding: 10px 15px;
      border-radius: 10px;
      position: relative;
      word-wrap: break-word;
    }
    
    .user-message {
      align-self: flex-end;
      background-color: var(--primary-color);
      color: white;
      margin-left: auto;
    }
    
    .ai-message {
      align-self: flex-start;
      background-color: white;
      border: 1px solid var(--border-color);
      margin-right: auto;
    }
    
    /* Example Questions Styling */
    .example-questions-container {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
      justify-content: center;
    }
    
    .example-question-btn {
      background-color: #2A2A2A;
      color: white;
      border: none;
      border-radius: 50px;
      padding: 10px 20px;
      font-size: 14px;
      cursor: pointer;
      transition: background-color 0.3s ease;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
      max-width: 300px;
    }
    
    .example-question-btn:hover {
      background-color: #404040;
    }
    
    /* Welcome container styling */
    .welcome-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      margin-bottom: 30px;
    }
    
    .typing-indicator {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 20px;
    }
    
    .typing-indicator span {
      height: 8px;
      width: 8px;
      background-color: #888;
      border-radius: 50%;
      display: inline-block;
      margin: 0 2px;
      animation: bounce 1.5s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(2) {
      animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
      animation-delay: 0.4s;
    }
    
    @keyframes bounce {
      0%, 60%, 100% {
        transform: translateY(0);
      }
      30% {
        transform: translateY(-5px);
      }
    }
    
    .model-dropdown {
      position: fixed;
      background-color: var(--bg-color);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      width: 250px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      z-index: 1000;
      margin-top: 5px;
      transition: background-color 0.3s ease;
      max-height: 400px;
      overflow-y: auto;
    }
    
    .provider-section {
      padding: 0;
      border-bottom: 1px solid var(--border-color);
    }
    
    .provider-section:last-child {
      border-bottom: none;
    }
    
    .provider-name {
      font-weight: 600;
      font-size: 12px;
      color: var(--text-color);
      padding: 8px 15px;
      background-color: rgba(0, 0, 0, 0.03);
    }
    
    [data-theme="dark"] .provider-name {
      background-color: rgba(255, 255, 255, 0.05);
    }
    
    .model-option {
      padding: 10px 15px;
      cursor: pointer;
      border-bottom: 1px solid var(--border-color);
      transition: background-color 0.3s ease;
    }
    
    .model-option:last-child {
      border-bottom: none;
    }
    
    .model-option:hover {
      background-color: var(--button-bg-hover);
    }
    
    .model-option.selected {
      background-color: rgba(24, 144, 255, 0.1);
    }
    
    .model-name {
      font-weight: 500;
      margin-bottom: 2px;
    }
    
    .model-description {
      font-size: 12px;
      color: #666;
    }
    
    [data-theme="dark"] .model-description {
      color: #aaa;
    }
    
    /* Focus mode styles */
    .app-container.focus-mode {
      grid-template-columns: 1fr !important;
    }
    
    .input-container.focus-mode {
      position: fixed;
      bottom: 0;
      left: 0;
      width: 100%;
      background-color: var(--bg-color);
      padding: 15px;
      box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
      z-index: 100;
    }
    
    .code-btn.active {
      background-color: var(--primary-color);
      color: white;
    }
    
    /* Animation for focus mode transition */
    .focus-mode {
      transition: all 0.3s ease;
    }
    
    /* Code execution interface styles */
    .code-execution-container {
      margin: 15px 0;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      background-color: var(--bg-color);
    }
    
    .code-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 15px;
      background-color: rgba(0, 0, 0, 0.05);
      border-bottom: 1px solid var(--border-color);
    }
    
    [data-theme="dark"] .code-header {
      background-color: rgba(255, 255, 255, 0.05);
    }
    
    .code-language {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
    }
    
    .code-actions {
      display: flex;
      gap: 8px;
    }
    
    .code-actions button {
      padding: 6px 12px;
      border: none;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .edit-btn {
      background-color: #17a2b8;
      color: white;
    }
    
    .edit-btn:hover {
      background-color: #138496;
    }
    
    .execute-btn {
      background-color: #28a745;
      color: white;
    }
    
    .execute-btn:hover {
      background-color: #218838;
    }
    
    .execute-btn:disabled {
      background-color: #6c757d;
      cursor: not-allowed;
    }
    
    .copy-btn {
      background-color: #6c757d;
      color: white;
    }
    
    .copy-btn:hover {
      background-color: #5a6268;
    }
    
    .code-display {
      margin: 0;
      background-color: #f8f9fa;
      border: none;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 14px;
      line-height: 1.5;
      overflow-x: auto;
    }
    
    [data-theme="dark"] .code-display {
      background-color: #2d3748;
    }
    
    .code-display code {
      display: block;
      padding: 15px;
      color: var(--text-color);
      white-space: pre;
    }
    
    .code-edit-container {
      padding: 0;
    }
    
    .code-edit-textarea {
      width: 100%;
      min-height: 200px;
      padding: 15px;
      border: none;
      outline: none;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 14px;
      line-height: 1.5;
      background-color: #f8f9fa;
      color: var(--text-color);
      resize: vertical;
      box-sizing: border-box;
    }
    
    [data-theme="dark"] .code-edit-textarea {
      background-color: #2d3748;
      color: #e2e8f0;
    }
    
    .edit-controls {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      padding: 10px 15px;
      background-color: rgba(0, 0, 0, 0.02);
      border-top: 1px solid var(--border-color);
    }
    
    [data-theme="dark"] .edit-controls {
      background-color: rgba(255, 255, 255, 0.02);
    }
    
    .edit-controls button {
      padding: 6px 12px;
      border: none;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .save-edit-btn {
      background-color: #28a745;
      color: white;
    }
    
    .save-edit-btn:hover {
      background-color: #218838;
    }
    
    .cancel-edit-btn {
      background-color: #dc3545;
      color: white;
    }
    
    .cancel-edit-btn:hover {
      background-color: #c82333;
    }
    
    .format-btn {
      background-color: #ffc107;
      color: #212529;
    }
    
    .format-btn:hover {
      background-color: #e0a800;
    }
    
    .code-output {
      border-top: 1px solid var(--border-color);
      background-color: var(--bg-color);
    }
    
    .code-error {
      padding: 15px;
      background-color: #f8d7da;
      color: #721c24;
      border-left: 4px solid #dc3545;
    }
    
    [data-theme="dark"] .code-error {
      background-color: #2c1618;
      color: #f5c6cb;
    }
    
    .code-success {
      padding: 15px;
      background-color: #d4edda;
      color: #155724;
      border-left: 4px solid #28a745;
    }
    
    [data-theme="dark"] .code-success {
      background-color: #0f2419;
      color: #b8dabd;
    }
    
    .code-text-output {
      padding: 15px;
      background-color: #f8f9fa;
      border-left: 4px solid #17a2b8;
    }
    
    [data-theme="dark"] .code-text-output {
      background-color: #2d3748;
    }
    
    .code-text-output pre {
      margin: 0;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 13px;
      line-height: 1.4;
      white-space: pre-wrap;
      color: var(--text-color);
    }
    
    .code-plot-output {
      padding: 15px;
      text-align: center;
      background-color: var(--bg-color);
    }
    
    /* Focus mode styles */
    .app-container.focus-mode {
      grid-template-columns: 1fr !important;
    }
    
    .input-container.focus-mode {
      position: fixed;
      bottom: 0;
      left: 0;
      width: 100%;
      background-color: var(--bg-color);
      padding: 15px;
      box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
      z-index: 100;
    }
    
    .code-btn.active {
      background-color: var(--primary-color);
      color: white;
    }
    
    /* Animation for focus mode transition */
    .focus-mode {
      transition: all 0.3s ease;
    }
    
    /* Responsive design for code interface */
    @media (max-width: 768px) {
      .code-actions {
        flex-direction: column;
        gap: 4px;
      }
      
      .code-actions button {
        font-size: 11px;
        padding: 4px 8px;
      }
      
      .edit-controls {
        flex-direction: column;
        gap: 4px;
      }
      
      .code-edit-textarea {
        font-size: 12px;
      }
    }
    
    /* Formatting warning styles */
    .formatting-warning {
      background-color: #fff3cd;
      border: 1px solid #ffeaa7;
      border-radius: 4px;
      margin: 10px 15px;
      animation: slideIn 0.3s ease;
    }
    
    [data-theme="dark"] .formatting-warning {
      background-color: #2c2416;
      border-color: #4a3f1a;
    }
    
    .warning-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 15px;
      border-bottom: 1px solid #ffeaa7;
      background-color: rgba(255, 193, 7, 0.1);
    }
    
    [data-theme="dark"] .warning-header {
      border-bottom-color: #4a3f1a;
      background-color: rgba(255, 193, 7, 0.05);
    }
    
    .warning-header i {
      color: #856404;
      margin-right: 8px;
    }
    
    [data-theme="dark"] .warning-header i {
      color: #ffc107;
    }
    
    .warning-header span {
      font-weight: 600;
      color: #856404;
      flex-grow: 1;
    }
    
    [data-theme="dark"] .warning-header span {
      color: #ffc107;
    }
    
    .auto-fix-btn {
      background-color: #ffc107;
      color: #212529;
      border: none;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .auto-fix-btn:hover {
      background-color: #e0a800;
    }
    
    .warning-list {
      margin: 0;
      padding: 10px 15px;
      list-style: none;
    }
    
    .warning-list li {
      color: #856404;
      font-size: 13px;
      margin-bottom: 5px;
      position: relative;
      padding-left: 20px;
    }
    
    [data-theme="dark"] .warning-list li {
      color: #ffc107;
    }
    
    .warning-list li:before {
      content: "•";
      position: absolute;
      left: 0;
      color: #ffc107;
      font-weight: bold;
    }
    
    .warning-list li:last-child {
      margin-bottom: 0;
    }
    
    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(-10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    /* Formatting error styles (more severe than warnings) */
    .formatting-error {
      background-color: #f8d7da;
      border: 1px solid #f5c6cb;
      border-radius: 4px;
      margin: 10px 15px;
      animation: slideIn 0.3s ease;
    }
    
    [data-theme="dark"] .formatting-error {
      background-color: #2c1618;
      border-color: #5a1e22;
    }
    
    .error-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 15px;
      border-bottom: 1px solid #f5c6cb;
      background-color: rgba(220, 53, 69, 0.1);
    }
    
    [data-theme="dark"] .error-header {
      border-bottom-color: #5a1e22;
      background-color: rgba(220, 53, 69, 0.05);
    }
    
    .error-header i {
      color: #721c24;
      margin-right: 8px;
    }
    
    [data-theme="dark"] .error-header i {
      color: #f5c6cb;
    }
    
    .error-header span {
      font-weight: 600;
      color: #721c24;
      flex-grow: 1;
    }
    
    [data-theme="dark"] .error-header span {
      color: #f5c6cb;
    }
    
    .warning-actions {
      display: flex;
      gap: 6px;
    }
    
    .validate-btn {
      background-color: #17a2b8;
      color: white;
      border: none;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .validate-btn:hover {
      background-color: #138496;
    }
    
    .suggestions-section {
      padding: 10px 15px;
      border-top: 1px solid #ffeaa7;
      background-color: rgba(255, 193, 7, 0.05);
    }
    
    [data-theme="dark"] .suggestions-section {
      border-top-color: #4a3f1a;
      background-color: rgba(255, 193, 7, 0.02);
    }
    
    .suggestions-section strong {
      color: #856404;
      display: block;
      margin-bottom: 8px;
    }
    
    [data-theme="dark"] .suggestions-section strong {
      color: #ffc107;
    }
    
    .suggestions-list {
      margin: 0;
      padding-left: 20px;
      list-style: none;
    }
    
    .suggestions-list li {
      color: #6c757d;
      font-size: 12px;
      margin-bottom: 4px;
      position: relative;
    }
    
    [data-theme="dark"] .suggestions-list li {
      color: #adb5bd;
    }
    
    .suggestions-list li:before {
      content: "💡";
      position: absolute;
      left: -20px;
    }
    
    .suggestions-list li:last-child {
      margin-bottom: 0;
    }
    
    /* Validation success styles */
    .validation-success {
      background-color: #d4edda;
      border: 1px solid #c3e6cb;
      border-radius: 4px;
      margin: 10px 15px;
      animation: slideIn 0.3s ease;
    }
    
    [data-theme="dark"] .validation-success {
      background-color: #0f2419;
      border-color: #1e4620;
    }
    
    .success-header {
      display: flex;
      align-items: center;
      padding: 10px 15px;
      background-color: rgba(40, 167, 69, 0.1);
    }
    
    [data-theme="dark"] .success-header {
      background-color: rgba(40, 167, 69, 0.05);
    }
    
    .success-header i {
      color: #155724;
      margin-right: 8px;
    }
    
    [data-theme="dark"] .success-header i {
      color: #b8dabd;
    }
    
    .success-header span {
      font-weight: 600;
      color: #155724;
    }
    
    [data-theme="dark"] .success-header span {
      color: #b8dabd;
    }
    
    /* Enhanced button states */
    .code-actions button:disabled,
    .edit-controls button:disabled,
    .warning-actions button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    
    /* Loading spinner animation */
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    .fa-spinner {
      animation: spin 1s linear infinite;
    }
  `;
  document.head.appendChild(style);
});

// Load model options from backend
function loadModelOptions() {
  const modelSelect = document.getElementById('model-select');
  modelSelect.innerHTML = '<option value="">Loading models...</option>';
  
  // Call the backend API to get available models
  fetch('/api/models')
    .then(response => response.json())
    .then(data => {
      modelSelect.innerHTML = '';
      
      // Group models by provider
      const modelsByProvider = {};
      data.models.forEach(model => {
        if (!modelsByProvider[model.provider]) {
          modelsByProvider[model.provider] = [];
        }
        modelsByProvider[model.provider].push(model);
      });
      
      // Add models to select dropdown grouped by provider
      Object.keys(modelsByProvider).forEach(provider => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = formatProviderName(provider);
        
        modelsByProvider[provider].forEach(model => {
          const option = document.createElement('option');
          option.value = model.id;
          option.textContent = `${formatModelName(model.name)} - ${model.description}`;
          option.dataset.provider = provider;
          option.dataset.name = model.name;
          optgroup.appendChild(option);
        });
        
        modelSelect.appendChild(optgroup);
      });
      
      // Select the first option by default
      if (modelSelect.options.length > 0) {
        modelSelect.selectedIndex = 0;
        modelSelect.dispatchEvent(new Event('change'));
      }
    })
    .catch(error => {
      console.error('Error loading models:', error);
      modelSelect.innerHTML = '<option value="">Failed to load models</option>';
    });
}

// Format provider name
function formatProviderName(provider) {
  const names = {
    'openai': 'OpenAI',
    'claude': 'Claude',
    'google': 'Google'
  };
  
  return names[provider.toLowerCase()] || provider.charAt(0).toUpperCase() + provider.slice(1);
}

// Check code for formatting issues and show warnings
function checkCodeFormatting(textarea) {
  const code = textarea.value;
  const container = textarea.closest('.code-execution-container');
  
  // Remove existing warnings
  const existingWarning = container.querySelector('.formatting-warning');
  if (existingWarning) {
    existingWarning.remove();
  }
  
  // Check for common formatting issues
  const issues = [];
  
  // Check for concatenated statements
  if (code.includes(');') && /\)[a-zA-Z_]/.test(code)) {
    issues.push("Declaraciones concatenadas detectadas - usa saltos de línea");
  }
  
  // Check for broken file paths
  if (/uploads\s*\/\s*[a-f0-9]+\s*-/.test(code)) {
    issues.push("Rutas de archivo con espacios detectadas");
  }
  
  // Check for broken operators
  if (/=\s*=\s*=/.test(code) && !/===/.test(code)) {
    issues.push("Operadores rotos detectados (= = =)");
  }
  
  // Check for missing imports at the beginning
  const lines = code.split('\n');
  const hasImports = lines.some(line => line.trim().startsWith('import ') || line.trim().startsWith('from '));
  const hasDataScienceCode = /\b(pd\.|np\.|plt\.|sns\.)/g.test(code);
  
  if (hasDataScienceCode && !hasImports) {
    issues.push("Faltan declaraciones de import para pandas, numpy, matplotlib");
  }
  
  // Check for broken comments
  if (/^\s*[A-Z][a-z]+\s+(de|del|por|para|con)/m.test(code) && !/^#/.test(code)) {
    issues.push("Comentarios sin # detectados");
  }
  
  // Show warning if issues found
  if (issues.length > 0) {
    showFormattingWarning(container, issues);
  }
}

// Show formatting warning with suggestions
function showFormattingWarning(container, issues) {
  const warning = document.createElement('div');
  warning.className = 'formatting-warning';
  warning.innerHTML = `
    <div class="warning-header">
      <i class="fas fa-exclamation-triangle"></i>
      <span>Problemas de formateo detectados</span>
      <button class="auto-fix-btn" onclick="autoFixFormatting(this)">
        <i class="fas fa-magic"></i> Auto-corregir
      </button>
    </div>
    <ul class="warning-list">
      ${issues.map(issue => `<li>${issue}</li>`).join('')}
    </ul>
  `;
  
  // Insert warning before edit controls
  const editControls = container.querySelector('.edit-controls');
  editControls.parentNode.insertBefore(warning, editControls);
}

// Auto-fix formatting issues
function autoFixFormatting(button) {
  const container = button.closest('.code-execution-container');
  const textarea = container.querySelector('.code-edit-textarea');
  
  // Apply comprehensive formatting
  let fixedCode = cleanPythonCode(textarea.value);
  
  // Additional fixes for common issues
  fixedCode = fixedCode
    // Fix missing imports
    .replace(/^(?!import|from|#)/m, (match, offset, string) => {
      const hasDataScienceCode = /\b(pd\.|np\.|plt\.|sns\.)/g.test(string);
      if (hasDataScienceCode && offset === 0) {
        return `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

${match}`;
      }
      return match;
    })
    // Fix broken file paths more aggressively
    .replace(/uploads\s*\/\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*\.\s*csv/g, 
             'uploads/$1-$2-$3-$4-$5.csv')
    // Fix concatenated print statements more thoroughly
    .replace(/(\))(\s*print\()/g, '$1\n$2')
    // Fix concatenated assignments
    .replace(/(\))(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=)/g, '$1\n$2')
    // Ensure proper line breaks after statements
    .replace(/(\))(\s*[a-zA-Z_])/g, '$1\n$2');
  
  textarea.value = fixedCode;
  adjustTextareaHeight(textarea);
  
  // Remove warning
  const warning = container.querySelector('.formatting-warning');
  if (warning) {
    warning.remove();
  }
  
  showVoiceIndicator('Código corregido automáticamente', 2000);
}