section .data
	msg db 'Hello, Guyssss', 0Ah

section .text
global _start

_start:
	mov ebx, msg			; eax e ebx sono entrambi puntati in memoria e puntano alla posizione di msg
	mov eax, ebx			; eax e ebx sono entrambi puntati in memoria e puntano alla posizione di msg

nextchar:
	cmp byte [eax], 0		; comparo i byte a cui eax punta, ovvero [eax], con 0, e serve e scrive 0 se sono diverse e 1 se sono uguali, nel flags
	jz finished				; salto a finished, se il cmp ha dato come risultato 1
	inc eax					; incrementa di uno eax ovvero punta al byte successivo
	jmp nextchar			; jump to nextchar, fino a quanto la condizione del cmp non è verificata

finished:
	sub eax, ebx			; sottrae l'indirrizzo in ebx da quello di eax
							; e dato che inizivano con lo stesso indirizzo e eax è stato incrementato, allora sottraendoli si trova a linghezza della stringa
	mov edx, eax			; sposto la lunghezza del msg in edx, in quanto è il regitry che viene controllato dalla sys call
	mov eax, 4
	mov ebx, 1
	mov ecx, msg
	int 80h

	mov eax, 1
	mov ebx, 0
	int 80h